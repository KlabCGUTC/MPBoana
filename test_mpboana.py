import unittest
from fastapi.testclient import TestClient
import sys
import os

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from my_project.app import app

# Constants for endpoints and expected messages
ROOT_ENDPOINT = "/"
SENTIMENT_ANALYSIS_ENDPOINT = "/analyze/sentiment"
EXPECTED_ROOT_MESSAGE = {"message": "Welcome to the MPBoana API!"}
VALID_SENTIMENT_PAYLOAD = {"text": "Today is a wonderful day!"}
INVALID_SENTIMENT_PAYLOAD = {"text": ""}
SENTIMENT_RESPONSE_KEY = "sentiment_result"


class TestMPBoanaAPI(unittest.TestCase):
    def setUp(self):
        """Set up the FastAPI test client."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test the root endpoint (/)"""
        response = self.client.get(ROOT_ENDPOINT)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), EXPECTED_ROOT_MESSAGE)

    def test_sentiment_analysis_endpoint(self):
        """Test the sentiment analysis endpoint (/analyze/sentiment)"""
        self._make_post_request_and_assert(
            SENTIMENT_ANALYSIS_ENDPOINT,
            VALID_SENTIMENT_PAYLOAD,
            200,
            SENTIMENT_RESPONSE_KEY
        )

    def test_sentiment_analysis_error_handling(self):
        """Test the sentiment analysis endpoint with invalid input"""
        self._make_post_request_and_assert_status_in(
            SENTIMENT_ANALYSIS_ENDPOINT,
            INVALID_SENTIMENT_PAYLOAD,
            [400, 500]
        )

    def _make_post_request_and_assert(self, endpoint, payload, expected_status, json_key):
        """Helper method to make POST request and assert response"""
        response = self.client.post(endpoint, json=payload)
        self.assertEqual(response.status_code, expected_status)
        self.assertIn(json_key, response.json())

    def _make_post_request_and_assert_status_in(self, endpoint, payload, expected_status_list):
        """Helper method to make POST request and assert status is in expected_status_list"""
        response = self.client.post(endpoint, json=payload)
        self.assertIn(response.status_code, expected_status_list)


if __name__ == "__main__":
    unittest.main()
