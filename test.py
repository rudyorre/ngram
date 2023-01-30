import unittest
import math
import language_model

class TestLanguageModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.train_data, self.dev_data = language_model.load_data()

    def test_ngram(self):
        lm = language_model.LanguageModel(self.train_data, 1, 0)
        freq_ngrams = sorted(lm.train_ngram_freq.items(), key=lambda x: x[1], reverse=True)[:1] 
        self.assertEqual(freq_ngrams, [(('the',), 68100)])
        lm = language_model.LanguageModel(self.train_data, 2, 0)
        freq_ngrams = sorted(lm.train_ngram_freq.items(), key=lambda x: x[1], reverse=True)[:1] 
        self.assertEqual(freq_ngrams, [(('<bos>', 'the'), 10893)])

    def test_basic_perplexity(self):
        lm = language_model.LanguageModel(self.train_data, 1, 0)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 1181.0737803820207)
        self.assertTrue(math.isnan(lm.corpus_perplexity(self.dev_data)))
        lm = language_model.LanguageModel(self.train_data, 2, 0)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 62.52893276577851)
        self.assertTrue(math.isnan(lm.corpus_perplexity(self.dev_data)))

    def test_smooth_perplexity(self):
        lm = language_model.LanguageModel(self.train_data, 1, 1)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 1187.9058009112703)
        self.assertAlmostEqual(lm.corpus_perplexity(self.dev_data), 1212.9720139207243)
        lm = language_model.LanguageModel(self.train_data, 2, 1)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 2272.8367100784803)
        self.assertAlmostEqual(lm.corpus_perplexity(self.dev_data), 2818.099515327075)

    def test_generate_sent(self):
        lm = language_model.LanguageModel(self.train_data, 1, 0)
        self.assertEqual(lm.greedy_search(), ' '.join(['the']*50))
        lm = language_model.LanguageModel(self.train_data, 2, 0)
        self.assertEqual(lm.greedy_search(), '<bos> the company said <eos>')
        

if __name__ == '__main__':
    unittest.main()
