import express from "express";
import OpenAI from "openai";

const app = express();
app.use(express.json({ limit: "1mb" }));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Prosty endpoint: dostaje fakty o portfelu i zwraca komentarz (bullet points)
app.post("/api/ai/commentary", async (req, res) => {
  try {
    const facts = req.body;

    const prompt = `
Jesteś analitykiem inwestycyjnym z wybitnymi zdolnościami w analizach portfelowych. 
Napisz krótki komentarz do wyników portfela, który skonfigurował użytkownik.
Zasady:
- język: polski
- 4–7 punktów w formie listy (każdy punkt w osobnej linii, zaczynając od "• ")
- bez porad inwestycyjnych typu "kup/sprzedaj"
- skup się na interpretacji: zwrot, ryzyko, Sharpe, koncentracja wag, korelacje, jakość danych
- weź pod uwagę korelacje i wagi aktywów
- oceń punktowo portfel według własnych kryteriów (np. 1-10) i uzasadnij ocenę
- UŻYWAJ TYLKO PODANYCH DANYCH, NIE WYMYSŁAJ LICZB ANI FAKTÓW
Wskaż co można by poprawić.
Dane (roczne, zannualizowane):
${JSON.stringify(facts, null, 2)}
`;

    const response = await client.chat.completions.create({
      model: "gpt-4",  // Use a valid model like "gpt-4" or "gpt-3.5-turbo"
      messages: [{ role: "user", content: prompt }],
      max_tokens: 500,
    });

    res.json({ text: response.choices[0].message.content });
  } catch (err) {
    res.status(500).json({ error: String(err?.message || err) });
  }
});

const port = process.env.PORT || 4173;
app.listen(port, () => console.log(`AI API listening on http://localhost:${port}`));
