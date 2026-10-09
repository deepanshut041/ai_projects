# Teaching and authoring route

```mermaid
flowchart TD
  Request[Requested artifact or lesson] --> Mode{Session mode}
  Mode -->|Author| Plan[Proposed prerequisites and dependency map]
  Mode -->|Live| Probe[One relevant question, wait for answer]
  Mode -->|Explain directly| Worked[Motivated worked explanation]
  Probe --> Plan
  Plan --> Step[Motivate and derive a step]
  Step --> Attempt[Prediction or transfer attempt]
  Attempt --> Feedback[Feedback after actual answer]
  Feedback -->|Adapt live route| Step
  Plan -->|Artifact| Render[Author, validate, render and inspect]
  Worked --> Optional[Optional practice]
```

These are authoring/session decisions, not model execution arrows. Artifact answers remain available behind reveals; live feedback requires an actual human reply. No learner replies or improved outcomes were observed here.
