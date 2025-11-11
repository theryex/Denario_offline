import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from denario import Denario

# Expected error message
EXPECTED_ERROR_MSG = "The selected model failed to generate a valid research idea in the required format. Please try a different, more capable model."

class TestErrorHandling(unittest.TestCase):

    def setUp(self):
        self.project_dir = "temp_test_project"
        self.den = Denario(project_dir=self.project_dir, clear_project_dir=True)
        self.den.set_data_description("Test description")

    def tearDown(self):
        shutil.rmtree(self.project_dir)

    @patch('denario.denario.build_lg_graph')
    @patch('denario.denario.Denario._create_llm_client')
    def test_get_idea_fast_handles_parsing_failure(self, mock_create_llm_client, mock_build_lg_graph):
        """
        Tests that get_idea_fast correctly handles the error marker from the graph.
        """
        print("--- Testing Graceful Error Handling via Mocking ---")

        # Mock the LLM client to avoid validation errors
        mock_create_llm_client.return_value = MagicMock()

        # Mock the graph to return a state containing the error marker
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            'idea': {'idea': 'ERROR::MODEL_OUTPUT_PARSE_FAILURE'}
        }
        mock_build_lg_graph.return_value = mock_graph

        # Use a dummy model config, as it won't be used by the mocked client
        dummy_model_config = {
            "provider": "vLLM", "api_base": "http://localhost:1234", "model": "dummy"
        }

        # This call should now handle the mocked error correctly
        self.den.get_idea_fast(llm=dummy_model_config)

        final_idea = self.den.research.idea
        print(f"Final Idea Content: '{final_idea}'")

        # Assert that the final idea is the user-friendly error message
        self.assertEqual(final_idea, EXPECTED_ERROR_MSG)
        print("--- Test Passed: Mocked error was handled gracefully. ---")

if __name__ == '__main__':
    unittest.main()
