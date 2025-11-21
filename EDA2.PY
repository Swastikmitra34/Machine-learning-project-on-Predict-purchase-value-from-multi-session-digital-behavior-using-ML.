import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv')

# Convert necessary columns to numeric
df['purchaseValue'] = pd.to_numeric(df['purchaseValue'], errors='coerce')
df['pageViews'] = pd.to_numeric(df['pageViews'], errors='coerce')
df['totalHits'] = pd.to_numeric(df['totalHits'], errors='coerce')
df['totals.bounces'] = pd.to_numeric(df['totals.bounces'], errors='coerce')

print("\n 1 Correlation between pageViews and purchaseValue (non-bounced users):")
non_bounced = df[(df['totals.bounces'] != 1) | df['totals.bounces'].isna()]
correlation = non_bounced[['pageViews', 'purchaseValue']].dropna().corr().iloc[0, 1]
print(f"Correlation: {correlation:.4f}")

print("\n 2 Continent with highest average totalHits:")
df['geoNetwork.continent'] = df['geoNetwork.continent'].astype(str)
continent_hits = df.groupby('geoNetwork.continent')['totalHits'].mean().sort_values(ascending=False)
print(continent_hits)
print(f"Top Continent: {continent_hits.idxmax()} ({continent_hits.max():.2f})")

print("\n 3 Most common trafficSource.medium among users who made a purchase:")
purchasers = df[df['purchaseValue'] > 0]
most_common_medium = purchasers['trafficSource.medium'].mode()[0]
print(f"Most common: {most_common_medium}")
print(purchasers['trafficSource.medium'].value_counts())

print("\n 4 Most common OS-browser combination among purchasers:")
combo_counts = purchasers.groupby(['os', 'browser']).size().sort_values(ascending=False)
print(combo_counts.head(5))
options = [
    ('Macintosh', 'Chrome'),
    ('Windows', 'Edge'),
    ('Andriod', 'Internet Explorer'),
    ('iOS', 'Internet Explorer')
]
print("\nGiven combinations:")
for combo in options:
    print(f"{combo}: {combo_counts.get(combo, 0)}")

print("\n 5️ Ratio of average totalHits: purchase vs no-purchase")
avg_hits_purchase = df[df['purchaseValue'] > 0]['totalHits'].mean()
avg_hits_no_purchase = df[df['purchaseValue'] == 0]['totalHits'].mean()
ratio = avg_hits_purchase / avg_hits_no_purchase
print(f"Ratio: {ratio:.4f}")

print("\n 6 Proportion of 'organic' sessions for each trafficSource.medium:")
df['trafficSource.medium'] = df['trafficSource.medium'].astype(str).str.lower()
total_medium = df['trafficSource.medium'].value_counts()
organic_medium = df[df['trafficSource.medium'] == 'organic']['trafficSource.medium'].value_counts()
proportion_df = (organic_medium / total_medium).fillna(0).sort_values(ascending=False)
summary = pd.DataFrame({
    'Total Sessions': total_medium,
    'Organic Sessions': organic_medium,
    'Proportion Organic': organic_medium / total_medium
}).fillna(0).sort_values(by='Proportion Organic', ascending=False)
print(summary)

print("\n 7 Most common referralPath among purchasers:")
referral_counts = purchasers['trafficSource.referralPath'].value_counts()
print(referral_counts.head(5))
options = ['/offer/2145', '/mail/u/1/', '/yt/about/', '/']
print("\nGiven options:")
for ref in options:
    print(f"{ref}: {referral_counts.get(ref, 0)}")

print("\n 8 OS with highest average purchase value:")
avg_purchase_by_os = df.groupby('os')['purchaseValue'].mean().sort_values(ascending=False)
print(avg_purchase_by_os)
print(f"Top OS: {avg_purchase_by_os.idxmax()} ({avg_purchase_by_os.max():.2f})")

print("\n 9 Average number of sessions per user:")
avg_sessions_per_user = df.groupby('userId')['sessionId'].nunique().mean()
print(f"Average sessions per user: {avg_sessions_per_user:.2f}")

print("\n 10 Number of unique users:")
unique_users = df['userId'].nunique()
print(f"Unique users: {unique_users}")
