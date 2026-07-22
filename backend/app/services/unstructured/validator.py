import re
from thefuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FaithfulnessValidator:
    def __init__(self):
        # A standard grammatical stopword list to filter out function words for Jaccard matching
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", 
            "been", "that", "this", "it", "its", "they", "them", "their", "he", 
            "him", "his", "she", "her", "we", "us", "our", "you", "your", "who", "which"
        }

    def calculate_jaccard(self, ground_truth: str, final_answer: str) -> float:
        """
        Calculates Jaccard containment (Intersection over Final Answer words)
        to prevent length-imbalance penalties from large databases.
        """
        gt_words = set(w for w in re.findall(r"\b\w+\b", ground_truth.lower()) if w not in self.stopwords)
        fa_words = set(w for w in re.findall(r"\b\w+\b", final_answer.lower()) if w not in self.stopwords)
        
        if not fa_words:
            return 0.0
            
        intersection = gt_words.intersection(fa_words)
        return len(intersection) / len(fa_words)

    def calculate_fuzzy(self, ground_truth: str, final_answer: str) -> float:
        """
        Calculates Fuzzy token set ratio (typo and rearrangement resistant overlap) using thefuzz.
        Returns a float between 0.0 and 100.0.
        """
        if not ground_truth.strip() or not final_answer.strip():
            return 0.0
        return float(fuzz.token_set_ratio(ground_truth, final_answer))

    def calculate_semantic(self, ground_truth: str, final_answer: str) -> float:
        """
        Calculates semantic similarity using CountVectorizer (Bag-of-Words) and cosine_similarity.
        This avoids the 2-document TF-IDF penalty on shared terms.
        Returns a float between 0.0 and 1.0.
        """
        if not ground_truth.strip() or not final_answer.strip():
            return 0.0
            
        try:
            vectorizer = CountVectorizer()
            counts = vectorizer.fit_transform([ground_truth, final_answer])
            sim = cosine_similarity(counts[0:1], counts[1:2])[0][0]
            return float(sim)
        except Exception:
            return 0.0

    def calculate_rouge_l(self, ground_truth: str, final_answer: str) -> float:
        """
        Calculates ROUGE-Lsum (sentence-level LCS recall) to prevent penalty on
        merged community sentence order rearrangements.
        Recall = Sum(LCS_length(sentence, ground_truth)) / len(final_answer_tokens)
        """
        # Split final answer into sentences to compute ROUGE-Lsum
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", final_answer) if s.strip()]
        gt_tokens = re.findall(r"\b\w+\b", ground_truth.lower())
        
        total_fa_tokens = len(re.findall(r"\b\w+\b", final_answer.lower()))
        if not gt_tokens or total_fa_tokens == 0:
            return 0.0
            
        total_lcs = 0
        for sentence in sentences:
            sen_tokens = re.findall(r"\b\w+\b", sentence.lower())
            if sen_tokens:
                lcs_len = self._calculate_lcs_length(gt_tokens, sen_tokens)
                total_lcs += lcs_len
                
        return total_lcs / total_fa_tokens

    def _calculate_lcs_length(self, x: list[str], y: list[str]) -> int:
        """
        Computes the Longest Common Subsequence (LCS) using an optimized O(N) space DP algorithm.
        """
        if len(x) < len(y):
            x, y = y, x
        m, n = len(x), len(y)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if x[i - 1] == y[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def verify_entities_and_numbers(self, ground_truth: str, final_answer: str) -> dict:
        """
        Extracts numbers (including quantities, dates, years) from final_answer,
        and verifies that they are present in the ground_truth database.
        """
        # Extract numbers using regex (e.g. 2026, 3.14, 100)
        fa_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", final_answer))
        gt_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", ground_truth))
        
        unverified_numbers = list(fa_numbers - gt_numbers)
        
        if not fa_numbers:
            score = 1.0
        else:
            verified_count = len(fa_numbers) - len(unverified_numbers)
            score = verified_count / len(fa_numbers)
            
        return {
            "score": score,
            "unverified_numbers": sorted(unverified_numbers),
            "unverified_entities": []  # Deprecated in favor of generic text similarity metrics
        }

    def translate_relationship(self, source: str, rel: str, target: str) -> str:
        """
        Generic relationship translator that formats raw edges into simple factual statements.
        Works across all domains and relationship labels without domain-specific rules.
        """
        rel_clean = rel.replace('_', ' ').strip().lower()
        return f"{source} {rel_clean} {target}."

    def clean_graph_syntax(self, text: str) -> str:
        """
        Translates raw graph database statements (nodes, edges, descriptions)
        into natural-language pseudo-sentences.
        """
        if not text:
            return ""
            
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 1. Translate Nodes: "Node: ID=xxx, Label=yyy, Type=zzz, Community=c, Description=desc"
            if line.startswith("Node:"):
                label_match = re.search(r'Label=([^,]+)', line)
                type_match = re.search(r'Type=([^,]+)', line)
                desc_match = re.search(r'Description=([^,\n\r]+)', line)
                
                label = label_match.group(1) if label_match else ""
                ntype = type_match.group(1) if type_match else ""
                desc = desc_match.group(1) if desc_match else ""
                
                if label:
                    node_str = f"{label}"
                    if ntype and ntype != "General":
                        node_str += f" ({ntype})"
                    if desc and desc.lower() != "none" and desc != "no description":
                        clean_desc = desc.split('.')[0].strip()
                        node_str += f" is {clean_desc}"
                    cleaned_lines.append(node_str + ".")
                continue
                
            # 2. Translate Edges using our smart translator
            if "-->" in line and "--[" in line:
                edge_match = re.search(r'(.+?)\s*--\[([\w_]+)\]-->\s*(.+)', line)
                if edge_match:
                    source = edge_match.group(1).strip()
                    rel = edge_match.group(2).strip()
                    target = edge_match.group(3).strip()
                    cleaned_lines.append(self.translate_relationship(source, rel, target))
                continue
                
            # 3. Keep community summaries and headers
            if line.startswith("Community ") or "Summary:" in line:
                cleaned_lines.append(line)
                
        cleaned_text = " ".join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def normalize_text_abbreviations(self, text: str) -> str:
        """
        Universal abbreviation normalizer (e.g. M.Sc. -> MSc, Ph.D. -> PhD, U.S.A. -> USA)
        to prevent token-splitting on internal punctuation, while preserving newlines.
        """
        if not text:
            return ""
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            words = line.split()
            normalized_words = []
            for w in words:
                clean_w = w.strip(".,;:!?()[]{}'\"")
                if '.' in clean_w and not clean_w.endswith('.'):
                    w = w.replace('.', '')
                elif clean_w.endswith('.') and '.' in clean_w[:-1]:
                    w = clean_w.replace('.', '') + '.'
                normalized_words.append(w)
            normalized_lines.append(" ".join(normalized_words))
        return "\n".join(normalized_lines)

    def evaluate(self, ground_truth: str, final_answer: str) -> dict:
        """
        Computes all faithfulness scores and checks them against strict logical thresholds:
        - Jaccard Containment Threshold: > 0.25
        - Fuzzy Threshold: > 65.0
        - Semantic Threshold: > 0.35
        - ROUGE-L Threshold: > 0.35
        - Entity/Numeric Threshold: > 0.85
        """
        # Universally normalize internal abbreviations
        norm_gt = self.normalize_text_abbreviations(ground_truth)
        norm_fa = self.normalize_text_abbreviations(final_answer)
        
        cleaned_gt = self.clean_graph_syntax(norm_gt)
        
        jaccard = self.calculate_jaccard(cleaned_gt, norm_fa)
        fuzzy = self.calculate_fuzzy(cleaned_gt, norm_fa)
        semantic = self.calculate_semantic(cleaned_gt, norm_fa)
        rouge_l = self.calculate_rouge_l(cleaned_gt, norm_fa)
        verification = self.verify_entities_and_numbers(cleaned_gt, norm_fa)
        
        j_pass = jaccard > 0.25
        f_pass = fuzzy > 65.0
        s_pass = semantic > 0.35
        r_pass = rouge_l > 0.35
        v_pass = verification["score"] > 0.85
        
        passed_count = sum([j_pass, f_pass, s_pass, r_pass, v_pass])
        
        if passed_count == 5:
            status = "Highly Faithful (Likely Correct)"
        elif passed_count >= 1:
            status = "Partial Match (Review Recommended)"
        else:
            status = "Hallucination Detected"
            
        return {
            "jaccard": jaccard,
            "fuzzy": fuzzy,
            "semantic": semantic,
            "rouge_l": rouge_l,
            "verification_score": verification["score"],
            "unverified_numbers": verification["unverified_numbers"],
            "unverified_entities": verification["unverified_entities"],
            "jaccard_pass": j_pass,
            "fuzzy_pass": f_pass,
            "semantic_pass": s_pass,
            "rouge_l_pass": r_pass,
            "verification_pass": v_pass,
            "status": status,
            "ground_truth": cleaned_gt,
            "final_answer": final_answer
        }
