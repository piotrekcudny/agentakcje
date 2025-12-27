import OpenAI from "openai";

export const config = {
  runtime: "nodejs",
};

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      res.status(500).json({ error: "Missing OPENAI_API_KEY" });
      return;
    }

    const client = new OpenAI({ apiKey });

    const facts = req.body;

    const prompt = `
Jesteś analitykiem portfelowym. Masz TYLKO dane z JSON poniżej.
Zasady:
- język: polski
- 4–7 punktów, każdy zaczyna się od "• "
- używaj wyłącznie liczb z JSON, nie wymyślaj
- bez porad "kup/sprzedaj"
JSON:
${JSON.stringify(facts)}
`;

    const response = await client.responses.create({
      model: "gpt-5-mini",
      input: prompt,
      temperature: 0.2,
    });

    res.status(200).json({ text: response.output_text });
  } catch (err: any) {
    const msg = err?.message ?? String(err);
    res.status(500).json({ error: msg });
  }
}
