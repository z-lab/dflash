import unittest

from dflash.edit_benchmark import MODULES, SOURCE, fenced, grade


class EditBenchTest(unittest.TestCase):
    def test_complete_file(self):
        answer = SOURCE.replace("feature_043 = 43\n", "feature_043 = 143\n")
        self.assertTrue(grade("single-file", fenced(answer, "python")))
        self.assertFalse(grade("single-file", answer.rstrip()))
        self.assertFalse(grade("single-file", SOURCE))

    def test_patch_application(self):
        blocks = [f"{name}\n<<<<<<< SEARCH\n{before}=======\n{before.replace('2', '4')}>>>>>>> REPLACE\n"
                  for name, before in MODULES.items()]
        self.assertTrue(grade("multi-file", "\n".join(blocks)))
        self.assertFalse(grade("multi-file", blocks[0]))
        self.assertFalse(grade("multi-file", blocks[0] + blocks[0] + blocks[1]))
        self.assertFalse(grade("multi-file", "\n".join(blocks).replace("policy.py", "../policy.py")))

    def test_sql(self):
        self.assertTrue(grade("new-code", "SELECT sensor, SUM(value) AS t FROM readings GROUP BY sensor "
                              "HAVING t > 0 ORDER BY t DESC, sensor ASC"))
        self.assertFalse(grade("new-code", "DELETE FROM readings"))
        self.assertFalse(grade("new-code", "SELECT load_extension('unused')"))
        self.assertFalse(grade("new-code", "SELECT sensor, SUM(value) FROM readings GROUP BY sensor"))
