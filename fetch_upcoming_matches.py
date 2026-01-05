"""
Fetch upcoming ATP matches from various sources.
This script retrieves scheduled matches for prediction purposes.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


class UpcomingMatchesFetcher:
    """Fetch upcoming ATP tennis matches from multiple sources."""

    def __init__(self):
        self.user_agent = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def fetch_from_sofascore(self) -> pd.DataFrame:
        """
        Fetch upcoming matches from SofaScore API (unofficial).
        Returns matches for the next 7 days.
        """
        print("📡 Fetching from SofaScore...")
        matches = []

        # SofaScore endpoint for ATP events
        base_url = "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events"

        try:
            # Get today's date
            today = datetime.now()

            for day_offset in range(7):  # Next 7 days
                date = today + timedelta(days=day_offset)
                date_str = date.strftime("%Y-%m-%d")

                url = f"{base_url}/{date_str}"
                response = requests.get(url, headers=self.user_agent, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    events = data.get("events", [])

                    for event in events:
                        # Filter for ATP events only
                        tournament = event.get("tournament", {})
                        category = tournament.get("category", {})

                        if category.get("name") == "ATP":
                            match_data = {
                                "date": date_str,
                                "tournament": tournament.get("name", ""),
                                "player_1": event.get("homeTeam", {}).get(
                                    "name", ""
                                ),
                                "player_2": event.get("awayTeam", {}).get(
                                    "name", ""
                                ),
                                "rank_1": event.get("homeTeam", {}).get("ranking"),
                                "rank_2": event.get("awayTeam", {}).get("ranking"),
                                "round": event.get("roundInfo", {}).get("name", ""),
                                "surface": tournament.get("surface", ""),
                                "series": category.get("slug", ""),
                                "status": event.get("status", {}).get("type", ""),
                            }
                            matches.append(match_data)

                    print(f"  ✓ {date_str}: Found {len(events)} events")
                else:
                    print(f"  ✗ {date_str}: Failed (status {response.status_code})")

        except Exception as e:
            print(f"❌ SofaScore fetch error: {e}")

        if matches:
            df = pd.DataFrame(matches)
            print(f"✅ SofaScore: Retrieved {len(df)} upcoming matches")
            return df
        else:
            print("⚠️ SofaScore: No matches found")
            return pd.DataFrame()

    def fetch_from_flashscore(self) -> pd.DataFrame:
        """
        Fetch from FlashScore (web scraping - backup source).
        Note: This is more fragile and may break if site structure changes.
        """
        print("📡 Fetching from FlashScore...")
        matches = []

        try:
            url = "https://www.flashscore.com/tennis/"
            response = requests.get(url, headers=self.user_agent, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Note: FlashScore uses JavaScript - this is a simplified example
                # You may need Selenium or Playwright for full functionality
                print("⚠️ FlashScore scraping requires JavaScript execution")
                print("   Consider using Selenium/Playwright for production")
            else:
                print(f"❌ FlashScore failed (status {response.status_code})")

        except Exception as e:
            print(f"❌ FlashScore error: {e}")

        return pd.DataFrame(matches)

    def fetch_from_tennis_data_uk(self) -> pd.DataFrame:
        """
        Fetch from Tennis-Data.co.uk (if they have an API endpoint).
        This is a placeholder - check their site for actual API.
        """
        print("📡 Checking Tennis-Data.co.uk...")
        print("⚠️ Tennis-Data.co.uk primarily provides historical data")
        return pd.DataFrame()

    def get_upcoming_matches(
        self, sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get upcoming matches from specified sources.

        Args:
            sources: List of sources to fetch from. Options: ['sofascore', 'flashscore']
                    If None, tries all sources.

        Returns:
            DataFrame with upcoming matches
        """
        if sources is None:
            sources = ["sofascore"]  # Default to most reliable source

        all_matches = []

        for source in sources:
            if source == "sofascore":
                df = self.fetch_from_sofascore()
            elif source == "flashscore":
                df = self.fetch_from_flashscore()
            elif source == "tennis_data":
                df = self.fetch_from_tennis_data_uk()
            else:
                print(f"⚠️ Unknown source: {source}")
                continue

            if not df.empty:
                df["source"] = source
                all_matches.append(df)

        if all_matches:
            combined_df = pd.concat(all_matches, ignore_index=True)
            # Remove duplicates (same match from multiple sources)
            combined_df = combined_df.drop_duplicates(
                subset=["date", "player_1", "player_2"], keep="first"
            )
            return combined_df
        else:
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filename: str = "upcoming_matches.csv"):
        """Save upcoming matches to CSV file."""
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"💾 Saved {len(df)} matches to {filename}")
        else:
            print("⚠️ No matches to save")


def main():
    """Main function to fetch and save upcoming matches."""
    print("🎾 ATP Upcoming Matches Fetcher\n")

    fetcher = UpcomingMatchesFetcher()

    # Fetch from available sources
    upcoming_df = fetcher.get_upcoming_matches(sources=["sofascore"])

    if not upcoming_df.empty:
        print(f"\n📋 Summary:")
        print(f"   Total matches: {len(upcoming_df)}")
        print(f"   Date range: {upcoming_df['date'].min()} to {upcoming_df['date'].max()}")
        print(f"   Tournaments: {upcoming_df['tournament'].nunique()}")

        # Display sample
        print("\n🎯 Sample upcoming matches:")
        print(
            upcoming_df[["date", "tournament", "player_1", "player_2", "round"]]
            .head(10)
            .to_string(index=False)
        )

        # Save to file
        fetcher.save_to_csv(upcoming_df)

        # Also save as JSON for easier consumption
        upcoming_df.to_json("upcoming_matches.json", orient="records", indent=2)
        print(f"💾 Also saved to upcoming_matches.json")

        return upcoming_df
    else:
        print("\n❌ No upcoming matches found!")
        print("   This could be due to:")
        print("   - Off-season period")
        print("   - API changes/restrictions")
        print("   - Network issues")
        return None


if __name__ == "__main__":
    result = main()
