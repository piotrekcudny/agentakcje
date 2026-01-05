import OpenAI from "openai";

// Klient zainicjalizowany raz poza handlerem (optymalizacja)
const client = new OpenAI(); 

export default async function handler(req, res) {
  // 1. Zabezpieczenie metody
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const facts = req.body;

    // 2. Zapytanie do OpenAI
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini", // Szybki i tani model
      messages: [
        { 
          role: "system", 
          content: "Jesteś ekspertem analizy portfelowej. Odpowiadaj w punktach po polsku." 
        },
        { 
          role: "user", 
          content: `Przeanalizuj te dane portfela: ${JSON.stringify(facts)}` 
        }
      ],
      max_tokens: 500,
    });

    // 3. Wysyłka odpowiedzi do frontendu
    return res.status(200).json({ text: response.choices[0].message.content });

  } catch (err) {
    console.error("Błąd serwera:", err);
    return res.status(500).json({ error: "Błąd podczas generowania komentarza." });
  }
}