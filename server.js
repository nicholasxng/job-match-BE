const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const cors = require('cors'); // Import the cors package

const app = express();
app.use(bodyParser.json());
app.use(cors()); // Enable CORS for all routes

const OPENAI_API_KEY = 'your-openai-api-key';

app.post('/match', async (req, res) => {
    const { resume, jobDescription } = req.body;

    const prompt = `Compare the following resume with the job description and provide an ATS match score along with a brief explanation:\n\nResume:\n${resume}\n\nJob Description:\n${jobDescription}`;

    try {
        const response = await axios.post('https://api.openai.com/v1/engines/davinci-codex/completions', {
            prompt,
            max_tokens: 500,
            temperature: 0.7,
        }, {
            headers: {
                'Authorization': `Bearer ${OPENAI_API_KEY}`,
                'Content-Type': 'application/json',
            },
        });

        res.json(response.data.choices[0].text.trim());
    } catch (error) {
        res.status(500).json({ error: 'Error processing request' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});