import unittest
import numpy as np
from retrieve import get_pipeline
from reason import generate_answer

class TestMemoryBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index, cls.model, cls.cases, cls.chunks = get_pipeline()
        cls.case_dict = {c["id"]: c for c in cls.cases}

    def test_retrieve(self):
        vec = self.model.encode(["client onboarding delays"], convert_to_numpy=True)[0]
        vec = vec.reshape(1, -1)
        _, indices = self.index.search(vec, 3)
        results = [self.cases[i] for i in indices[0]]
        self.assertGreater(len(results), 0)

    def test_reasoning(self):
        results = [{"metadata": {"id": "C001", "title": "Client Onboarding Slowdown", "context": "test", "outcome": "test", "lessons": "test", "tags": ["compliance"], "type": "legal"}, "similarity": 0.9}]
        out = generate_answer("compliance check", results)
        self.assertIn("compliance", " ".join(out["recommendations"]).lower())
        self.assertGreaterEqual(len(out["cases"]), 1)

if __name__ == "__main__":
    unittest.main()
