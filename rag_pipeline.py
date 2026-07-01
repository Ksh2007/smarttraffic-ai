from phase2_ir import retrieve_documents
from phase3_ml import predict_congestion, FEATURES
def build_rag_prompt(
    user_query: str,
    road:       str,
    speed:      int,
    count:      int,
    weather:    str,
    artifacts:  dict,
    top_k:      int = 3,
) -> tuple[str, str, str]:
    """
    Returns (prompt, predicted_congestion, retrieved_context).

    Step 1 — IR  : retrieve top-k relevant documents
    Step 2 — ML  : predict congestion from live conditions
    Step 3 — RAG : assemble grounded prompt for LLM
    """
    results = retrieve_documents(
        user_query,
        artifacts["corpus"],
        artifacts["vectorizer"],
        artifacts["tfidf_matrix"],
        top_k=top_k,
    )
    context = "\n".join(
        f"[{row.source.upper()}] {row.text}"
        for _, row in results.iterrows()
    )
    prediction = predict_congestion(
        road=road, speed=speed, count=count, weather=weather,
        dt_model=artifacts["dt_model"],
        le_road=artifacts["le_road"],
        le_weather=artifacts["le_weather"],
        le_label=artifacts["le_label"],
    )
    prompt = (
        "You are SmartTraffic AI, an intelligent assistant "
        "for urban traffic management.\n\n"
        f"User query: {user_query}\n\n"
        "Current live conditions:\n"
        f"  Road    : {road}\n"
        f"  Speed   : {speed} km/h\n"
        f"  Vehicles: {count} per hour\n"
        f"  Weather : {weather}\n"
        f"  ML predicted congestion level: {prediction}\n\n"
        "Relevant records retrieved from the database:\n"
        f"{context}\n\n"
        "Using the live conditions and retrieved records above, "
        "answer the user query accurately and concisely."
    )
    return prompt, prediction, context
def run_demo(artifacts):
    """Run 3 test queries to verify the pipeline."""
    test_cases = [
        {
            "query":   "Are there any accidents near Ring Road?",
            "road":    "Ring Road",
            "speed":   8,
            "count":   500,
            "weather": "Rainy",
        },
        {
            "query":   "Which road has the highest traffic volume?",
            "road":    "NH-24",
            "speed":   20,
            "count":   350,
            "weather": "Clear",
        },
        {
            "query":   "Show me signal failure reports",
            "road":    "MG Road",
            "speed":   12,
            "count":   420,
            "weather": "Foggy",
        },
    ]
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"  Query {i}: {case['query']}")
        print("="*60)
        prompt, prediction, context = build_rag_prompt(
            user_query=case["query"],
            road=case["road"],
            speed=case["speed"],
            count=case["count"],
            weather=case["weather"],
            artifacts=artifacts,
        )
        print(f"\n[ML Prediction] Congestion → {prediction}")
        print(f"\n[Retrieved Context]\n{context}")
        print(f"\n[RAG Prompt]\n{prompt}")
if __name__ == "__main__":
    from rag_loader import load_all_artifacts
    artifacts = load_all_artifacts()
    run_demo(artifacts)