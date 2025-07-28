import pandas as pd
import matplotlib.pyplot as plt
import os

# Change as needed:
LOGDIR = "logs"
EVERYTHING = os.path.join(LOGDIR, "everything.csv")
CANDIDATES = os.path.join(LOGDIR, "all_candidates.csv")
ROUNDS = os.path.join(LOGDIR, "round_metrics.csv")

# Load all outputs
df_main = pd.read_csv(EVERYTHING)
df_cand = pd.read_csv(CANDIDATES)
df_rounds = pd.read_csv(ROUNDS)

print("\n== Main Results ==")
print(df_main.head())

print("\n== All Candidates (first 10) ==")
print(df_cand.head(10))

print("\n== Round Metrics ==")
print(df_rounds.head())

# Plot round metric improvement
plt.figure(figsize=(8,4))
plt.plot(df_rounds['round'], df_rounds['delta_metric'], marker='o')
plt.title("Delta Metric per Round")
plt.xlabel("Round")
plt.ylabel("Delta Metric")
plt.grid(True)
plt.tight_layout()
plt.show()

# Inspect a few accepted edits for quality check
print("\n== Sample Accepted Edits ==")
accepted = df_cand[df_cand['accepted'] == 1]
print(accepted[['round','question','original_answer','candidate','f1','semantic_sim']].sample(5, random_state=42))

# Summary statistics
print("\n=== Accepted Candidates Stats ===")
print(accepted.describe())

# Distribution of semantic similarity for accepted vs. not accepted
plt.figure(figsize=(8,4))
accepted['semantic_sim'].hist(alpha=0.5, label="Accepted", bins=30)
df_cand[df_cand['accepted']==0]['semantic_sim'].hist(alpha=0.5, label="Rejected", bins=30)
plt.legend()
plt.title("Semantic Similarity: Accepted vs. Rejected")
plt.xlabel("Semantic Similarity")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
