from Services.QueryRefinementService.query_refinement_service import QueryRefinementService


def run_test():
    service = QueryRefinementService()

    queries = [
        "how to learn pythom programing",   # two misspellings
        "best laptop for students",          # clean -> synonym expansion
        "car insurance",                     # synonyms (automobile, etc.)
    ]

    for query in queries:
        result = service.refine(query, correct=True, expand=True, max_synonyms_per_word=2)
        print("=" * 70)
        print(f"Original : {result.original}")
        print(f"Refined  : {result.refined}")
        print(f"Spelling : {result.corrections}")
        print(f"Synonyms : {result.added_synonyms}")


if __name__ == "__main__":
    run_test()
