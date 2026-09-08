# Demo walkthrough

Generated 2026-09-08 05:58 UTC by `scripts/demo_walkthrough.py`, running the real application code against the trained production model.

**Parsing/response mode for this run:** rule-based fallback (no LLM API key configured)

## 1. A complete natural-language query

**User:** A 29-year-old woman travelling in first class with her husband, paid a $150 fare.

**Status:** `prediction`

**Response:** Based on the historical data, this passenger would likely have survived (estimated survival probability: 98.9%). This reflects patterns such as sex, passenger class, age, and fare in the training data -- it's a statistical estimate from a century-old dataset, not a certainty.

**Parsed fields:**
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29.0,
  "sibsp": 1,
  "parch": null,
  "fare": 150.0,
  "embarked": null,
  "out_of_scope": false,
  "ambiguous_fields": [],
  "source": "rules"
}
```

**Prediction:** Survived  |  **Survival probability:** 98.9%

**Model input features:**
```json
{
  "Age": 29.0,
  "Fare": 150.0,
  "FamilySize": 2,
  "Pclass": 1,
  "Sex": "female",
  "Embarked": "S",
  "Title": "Mrs",
  "IsAlone": 0,
  "Deck": "Unknown"
}
```

## 2. An out-of-scope query (edge case)

**User:** What's the weather like in Denver today?

**Status:** `out_of_scope`

**Response:** I'm a focused tool for one thing: estimating whether a Titanic passenger, described in a sentence, would have survived the sinking. I can't help with unrelated questions -- try describing a passenger instead, e.g. "a 22-year-old third-class male passenger travelling alone".

**Parsed fields:**
```json
{
  "pclass": null,
  "sex": null,
  "age": null,
  "sibsp": null,
  "parch": null,
  "fare": null,
  "embarked": null,
  "out_of_scope": true,
  "ambiguous_fields": [],
  "source": "rules"
}
```

## 3. An incomplete query -- missing required fields (edge case)

**User:** A passenger who paid $50.

**Status:** `needs_clarification`

**Response:** I need a bit more information to make a prediction: which class they were travelling in (1st, 2nd, or 3rd); their sex (male or female); their age. For example: "a 29-year-old woman in first class travelling with her husband".

**Parsed fields:**
```json
{
  "pclass": null,
  "sex": null,
  "age": null,
  "sibsp": null,
  "parch": null,
  "fare": 50.0,
  "embarked": null,
  "out_of_scope": false,
  "ambiguous_fields": [],
  "source": "rules"
}
```

## 4. Following up with the missing details

**User:** He was a 40-year-old male in third class.

**Status:** `prediction`

**Response:** Based on the historical data, this passenger would likely NOT have survived (estimated survival probability: 22.2%). This reflects patterns such as sex, passenger class, age, and fare in the training data -- it's a statistical estimate from a century-old dataset, not a certainty.

**Parsed fields:**
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 40.0,
  "sibsp": null,
  "parch": null,
  "fare": null,
  "embarked": null,
  "out_of_scope": false,
  "ambiguous_fields": [],
  "source": "rules"
}
```

**Prediction:** Did not survive  |  **Survival probability:** 22.2%

**Model input features:**
```json
{
  "Age": 40.0,
  "Fare": 8.05,
  "FamilySize": 1,
  "Pclass": 3,
  "Sex": "male",
  "Embarked": "S",
  "Title": "Mr",
  "IsAlone": 1,
  "Deck": "Unknown"
}
```
