#!/usr/bin/env bash

# check availability of GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
  echo "GEMINI_API_KEY is not set"
  exit 1
fi

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" \
  -H 'Content-Type: application/json' \
  -H "X-goog-api-key: $GEMINI_API_KEY" \
  -X POST \
  -d '{
    "contents": [
      {
        "parts": [
          {
            "text": "Explain how AI works as a short article. Format the output in markdown."
          }
        ]
      }
    ]
  }'