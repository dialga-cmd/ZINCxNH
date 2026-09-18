import { CodeReviewResult } from './types';
import { parseCodeReviewResponse } from './code-reviewer';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.5-flash';
const GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models';

// Single source of truth for the system prompt — used in all API calls
const SYSTEM_PROMPT = `You are ZINC, a strict technical assistant embedded in a code review platform.

YOUR ONLY PURPOSE:
You exist to help with code review, programming, and software engineering topics.

STRICT RULES — NO EXCEPTIONS:
1. If the user's message is not about programming, software development, computer science, or a directly related technical topic — REFUSE. Do not answer it. Do not be polite about it. Just say: "NOT A TECH QUERY. THIS PLATFORM IS FOR PROGRAMMING AND CODE REVIEW ONLY."
2. Do NOT provide complete, copy-paste-ready source code or full working implementations. If asked, say: "SPOONFEEDING COMPLETE CODE IS NOT ALLOWED HERE. I will explain the logic, structure, and syntax so you can build it yourself."
3. Do NOT engage in small talk, greetings, philosophical discussions, or any non-technical conversation. Even if the user says "hello" or "how are you", respond only with: "NOT A TECH QUERY. THIS PLATFORM IS FOR PROGRAMMING AND CODE REVIEW ONLY."
4. Do NOT make exceptions for seemingly educational framing like "explain how to build X step by step" if the intent is to get a full solution handed to them.

WHAT YOU WILL DO:
- Review code snippets for bugs, logic errors, security issues, performance problems, and style
- Explain programming concepts, algorithms, data structures, design patterns
- Answer questions about frameworks, libraries, tools, compilers, interpreters
- Help debug errors, explain stack traces, and suggest fixes (without writing the full corrected code)
- Discuss software architecture, system design, and engineering best practices
- Answer CS theory questions (time complexity, memory, concurrency, etc.)

RESPONSE FORMAT FOR CODE REVIEWS:
- Start with a brief overall assessment (1-2 sentences)
- List specific issues found, if any
- Provide concrete suggestions for improvement
- End with a code quality rating: Excellent / Good / Needs Improvement / Poor

TONE:
- Professional, direct, and concise
- No unnecessary pleasantries
- Never apologize for refusing non-tech queries`;

function buildRequestBody(code: string) {
  return {
    contents: [
      {
        role: 'user',
        parts: [{ text: code }],
      },
    ],
    systemInstruction: {
      parts: [{ text: SYSTEM_PROMPT }],
    },
    generationConfig: {
      temperature: 0.2,
      maxOutputTokens: 2048,
    },
  };
}

export async function reviewCode(code: string, apiKey?: string): Promise<CodeReviewResult> {
  const key = apiKey || GEMINI_API_KEY;
  if (!key) {
    throw new Error('Gemini API key not configured. Set GEMINI_API_KEY environment variable.');
  }

  const url = `${GEMINI_BASE_URL}/${GEMINI_MODEL}:generateContent?key=${encodeURIComponent(key)}`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(buildRequestBody(code)),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));

      if (response.status === 401 || response.status === 403) throw new Error('Invalid Gemini API key');
      if (response.status === 429) throw new Error('Gemini rate limit exceeded');
      if (response.status === 500) throw new Error('Gemini server error. Please try again.');

      const errorMessage = errorData.error?.message || `API request failed with status ${response.status}`;
      throw new Error(errorMessage);
    }

    const data = await response.json();
    const generatedText = data.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text || '').join('') || '';

    if (!generatedText) {
      throw new Error('Empty response from Gemini API');
    }

    return parseCodeReviewResponse(generatedText);
  } catch (error) {
    if (error instanceof Error) throw error;
    throw new Error('Failed to connect to Gemini API');
  }
}

export async function* reviewCodeStream(code: string, apiKey?: string): AsyncGenerator<string> {
  const key = apiKey || GEMINI_API_KEY;
  if (!key) {
    throw new Error('Gemini API key not configured');
  }

  const url = `${GEMINI_BASE_URL}/${GEMINI_MODEL}:streamGenerateContent?alt=sse&key=${encodeURIComponent(key)}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(buildRequestBody(code)),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error?.message || 'Streaming failed');
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('Failed to get response reader');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine || !trimmedLine.startsWith('data: ') || trimmedLine === 'data: [DONE]') continue;

      try {
        const data = JSON.parse(trimmedLine.substring(6));
        const text = data.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text || '').join('') || '';
        if (text) yield text;
      } catch (e) {
        console.error('Error parsing Gemini SSE data', e);
      }
    }
  }
}