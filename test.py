import unittest
from your_project.resume_parser import parse_resume
from your_project.resume_ranker import rank_resumes

class TestResumeRanker(unittest.TestCase):
    
    def setUp(self):
        """Setup test data."""
        self.resume1 = """John Doe\nExperience: 5 years in Data Science\nSkills: Python, Machine Learning"""
        self.resume2 = """Jane Smith\nExperience: 3 years in Web Development\nSkills: HTML, CSS, JavaScript"""
        
        self.job_description = """Looking for a Data Scientist with Python and Machine Learning experience."""
    
    def test_parse_resume(self):
        """Test if resume parsing extracts correct information."""
        parsed_resume = parse_resume(self.resume1)
        self.assertIn("John Doe", parsed_resume["name"])
        self.assertIn("Python", parsed_resume["skills"])
    
    def test_rank_resumes(self):
        """Test if resumes are ranked correctly based on job description."""
        resumes = [self.resume1, self.resume2]
        rankings = rank_resumes(resumes, self.job_description)
        
        self.assertGreater(rankings[0]["score"], rankings[1]["score"], "Data Science resume should rank higher.")
        
if __name__ == '__main__':
    unittest.main()
