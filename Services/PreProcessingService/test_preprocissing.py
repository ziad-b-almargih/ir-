from Services.PreProcessingService.preprocessing_service import PreprocessingService

if __name__ == "__main__":
    service = PreprocessingService()

    raw_texts = [
        "The system architecture handles complex SQL queries perfectly.",
        "Optimizing complex SQL queries improves the database architecture.",
        "A good database architecture handles both simple and complex queries.",
        "Information Retrieval systems rely on TF-IDF for document ranking.",
        "The new search engine uses BM25 and TF-IDF for better document ranking.",
        "Advanced document ranking relies on BM25 rather than just TF-IDF models.",
        "Building microservices ensures scalability and clean system architecture.",
        "An API gateway connects microservices to improve the overall architecture."
    ]

    for index, text in enumerate(raw_texts, start=1):
        print(f"\n[Text {index}] Original : {text}")
        print(f"[Text {index}] Processed: {service.process_text(text)}")
