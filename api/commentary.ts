import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.VITE_OPENAI_API_KEY });

export default async function handler(req, res) {
  // Vercel Serverless Functions obsługują tylko metody określone przez Ciebie
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

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

Dane (roczne, zannualizowane znajdziesz w załączonych plikach
${JSON.stringify(facts, null, 2)}
`;

    const response = await client.chat.completions.create({
      model: "gpt-5.1-mini", // Zalecam gpt-4o-mini - jest tańszy i szybszy
      messages: [{ role: "user", content: prompt }],
      max_tokens: 500,
    });

    return res.status(200).json({ text: response.choices[0].message.content });
  } catch (err) {
    return res.status(500).json({ error: String(err?.message || err) });
  }
}