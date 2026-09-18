import { NextRequest, NextResponse } from 'next/server';
import { CodeReviewResult } from '@/lib/ai-providers/types';
import { SYSTEM_PROMPT, buildCodeReviewPrompt, parseCodeReviewResponse } from '@/lib/ai-providers/code-reviewer';
import { getCachedReview, setCachedReview } from '@/lib/storage/cache-storage';
import { checkAndIncrementRateLimit } from '@/lib/rate-limit-server';

// Server-side API keys (not exposed to client)
const GEMINI_API_KEYS = (process.env.GEMINI_API_KEY || '').split(',').filter(k => k.trim());
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.5-flash';
const HF_API_KEYS = (process.env.HF_API_KEY || '').split(',').filter(k => k.trim());
const HF_MODELS = [
  'Qwen/Qwen2.5-Coder-32B-Instruct',
  'bigcode/starcoder2-15b-instruct-v0.1',
  'meta-llama/Llama-3.2-3B-Instruct'
];

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const HF_API_URL_PREFIX = 'https://api-inference.huggingface.co/models/';

// Rate limit tracking (in-memory for server instance)
const rateLimits = new Map<string, { until: number; count: number }>();

function isRateLimited(provider: string): boolean {
  const limit = rateLimits.get(provider);
  if (!limit) return false;
  if (Date.now() > limit.until) {
    rateLimits.delete(provider);
    return false;
  }
  return true;
}

function recordRateLimit(provider: string, retryAfterSeconds: number = 60) {
  const existing = rateLimits.get(provider);
  rateLimits.set(provider, {
    until: Date.now() + retryAfterSeconds * 1000,
    count: (existing?.count || 0) + 1,
  });
}

async function tryHuggingFace(code: string, apiKey: string, model: string): Promise<CodeReviewResult> {
  if (isRateLimited(`hf-${model}`)) throw new Error(`HF model ${model} rate limited`);

  // Use buildCodeReviewPrompt which embeds SYSTEM_PROMPT
  const prompt = buildCodeReviewPrompt(code);

  const response = await fetch(`${HF_API_URL_PREFIX}${model}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      inputs: prompt,
      parameters: { temperature: 0.2, max_new_tokens: 2048 },
      options: { wait_for_model: true }
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    if (response.status === 429) {
      recordRateLimit(`hf-${model}`, 60);
      throw new Error('HuggingFace rate limit exceeded');
    }
    throw new Error(data.error || `HuggingFace failed: ${response.status}`);
  }

  const data = await response.json();
  let text = '';
  if (Array.isArray(data)) {
    text = data[0]?.generated_text || data[0]?.content || '';
  } else {
    text = data.generated_text || data.content || '';
  }

  if (!text) throw new Error('Empty HuggingFace response');

  // Some HF models include the prompt in output, strip it
  if (text.includes('###')) {
    text = text.split('###').pop() || text;
  }

  return parseCodeReviewResponse(text);
}

async function tryGemini(code: string, apiKey: string): Promise<CodeReviewResult> {
  const key = encodeURIComponent(apiKey.trim());
  const response = await fetch(`${GEMINI_API_URL}/${GEMINI_MODEL}:generateContent?key=${key}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      contents: [
        {
          role: 'user',
          parts: [{ text: code }],
        },
      ],
      systemInstruction: {
        // Single source of truth — imported from code-reviewer.ts
        parts: [{ text: SYSTEM_PROMPT }],
      },
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: 2048,
      },
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    if (response.status === 429) {
      recordRateLimit('gemini', 60);
      throw new Error('Gemini rate limit exceeded');
    }
    throw new Error(data.error?.message || `Gemini failed: ${response.status}`);
  }

  const data = await response.json();
  const text = data.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text || '').join('') || '';
  if (!text) throw new Error('Empty Gemini response');
  return parseCodeReviewResponse(text);
}

export async function POST(request: NextRequest) {
  try {
    const { code } = await request.json();
    if (!code || typeof code !== 'string') {
      return NextResponse.json({ error: 'Code is required' }, { status: 400 });
    }

    const userId = request.headers.get('x-user-id');
    if (!userId) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const rateLimitResult = await checkAndIncrementRateLimit(userId);
    if (!rateLimitResult.allowed) {
      return NextResponse.json(
        {
          error: rateLimitResult.error || 'Rate limit exceeded',
          remaining: rateLimitResult.remaining,
          resetAt: rateLimitResult.resetAt?.toISOString(),
        },
        { status: 429 }
      );
    }

    const prompt = buildCodeReviewPrompt(code);

    // Check cache first
    const cached = await getCachedReview(prompt);
    if (cached) {
      console.log('CACHE_HIT // Server cache');
      return NextResponse.json({ ...cached, quota: rateLimitResult.remaining });
    }

    const errors: string[] = [];

    // Try Gemini first (multiple keys for rotation)
    for (const key of GEMINI_API_KEYS) {
      try {
        const result = await tryGemini(code, key.trim());
        await setCachedReview(prompt, result);
        return NextResponse.json({ ...result, quota: rateLimitResult.remaining });
      } catch (e) {
        errors.push(`Gemini: ${e instanceof Error ? e.message : 'failed'}`);
      }
    }

    // Try Hugging Face as backup
    for (const hfKey of HF_API_KEYS) {
      for (const model of HF_MODELS) {
        try {
          const result = await tryHuggingFace(code, hfKey.trim(), model);
          await setCachedReview(prompt, result);
          return NextResponse.json({ ...result, quota: rateLimitResult.remaining });
        } catch (e) {
          errors.push(`HF(${model}): ${e instanceof Error ? e.message : 'failed'}`);
        }
      }
    }

    return NextResponse.json({ error: `All providers failed: ${errors.join(' | ')}` }, { status: 503 });
  } catch (error) {
    console.error('[API/REVIEW] Error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}