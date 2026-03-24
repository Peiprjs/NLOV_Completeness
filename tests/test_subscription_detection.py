"""Tests for automatic subscription detection from Product column."""

import pandas as pd
import pytest

from app import (
    SUBSCRIPTION_OPTIONS,
    apply_subscriptions_to_merged_trips,
    detect_subscription_from_product,
    merge_check_in_out_transactions,
    normalize_subscription_state,
)


class TestSubscriptionDetection:
    """Test suite for detect_subscription_from_product function."""

    def test_detect_student_week_subscription(self):
        """Test detection of Student Week subscription."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD123"] * 3,
            "Product": ["Student Week Vrij (2e klas)"] * 3
        })
        
        result = detect_subscription_from_product(df, "CARD123")
        assert result == "Student Week (Free weekdays, Discount weekends)"

    def test_detect_student_weekend_subscription(self):
        """Test detection of Student Weekend subscription."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD456"] * 2,
            "Product": ["Student Weekend Vrij"] * 2
        })
        
        result = detect_subscription_from_product(df, "CARD456")
        assert result == "Student Weekend (Discount weekdays, Free weekends)"

    def test_detect_other_subscription(self):
        """Test detection of other subscription types."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD789"],
            "Product": ["NS Flex Weekend Vrij"]
        })
        
        result = detect_subscription_from_product(df, "CARD789")
        assert result == "Other Subscription"

    def test_no_subscription_for_empty_product(self):
        """Test that empty product returns No Subscription."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD000"],
            "Product": [""]
        })
        
        result = detect_subscription_from_product(df, "CARD000")
        assert result == "No Subscription"

    def test_no_subscription_for_missing_product(self):
        """Test that missing product column returns No Subscription."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD111"],
            "Datum": ["2024-01-01"]
        })
        
        result = detect_subscription_from_product(df, "CARD111")
        assert result == "No Subscription"

    def test_case_insensitive_detection(self):
        """Test that detection works regardless of case."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD222"] * 2,
            "Product": ["STUDENT WEEK VRIJ", "student week vrij"]
        })
        
        result = detect_subscription_from_product(df, "CARD222")
        assert result == "Student Week (Free weekdays, Discount weekends)"

    def test_non_existent_card_returns_default(self):
        """Test that non-existent card number returns default subscription."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD123"],
            "Product": ["Student Week Vrij"]
        })
        
        result = detect_subscription_from_product(df, "NONEXISTENT")
        assert result == "No Subscription"

    def test_empty_dataframe_returns_default(self):
        """Test that empty DataFrame returns default subscription."""
        df = pd.DataFrame()
        
        result = detect_subscription_from_product(df, "CARD123")
        assert result == "No Subscription"

    def test_student_week_prioritized_over_weekend(self):
        """Test that Student Week is selected when more common than Weekend."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD333"] * 5,
            "Product": [
                "Student Week Vrij",
                "Student Week Vrij",
                "Student Week Vrij",
                "Student Weekend Vrij",
                "Student Weekend Vrij"
            ]
        })
        
        result = detect_subscription_from_product(df, "CARD333")
        assert result == "Student Week (Free weekdays, Discount weekends)"

    def test_student_weekend_when_more_common(self):
        """Test that Student Weekend is selected when more common."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD444"] * 5,
            "Product": [
                "Student Week Vrij",
                "Student Weekend Vrij",
                "Student Weekend Vrij",
                "Student Weekend Vrij",
                "Student Weekend Vrij"
            ]
        })
        
        result = detect_subscription_from_product(df, "CARD444")
        assert result == "Student Weekend (Discount weekdays, Free weekends)"

    def test_multiple_cards_in_dataframe(self):
        """Test that detection works correctly with multiple cards in same DataFrame."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD1", "CARD1", "CARD2", "CARD2"],
            "Product": [
                "Student Week Vrij",
                "Student Week Vrij",
                "Student Weekend Vrij",
                "Student Weekend Vrij"
            ]
        })
        
        card1_result = detect_subscription_from_product(df, "CARD1")
        card2_result = detect_subscription_from_product(df, "CARD2")
        
        assert card1_result == "Student Week (Free weekdays, Discount weekends)"
        assert card2_result == "Student Weekend (Discount weekdays, Free weekends)"

    def test_with_real_ovchipkaart_data_format(self):
        """Test with realistic OV-Chipkaart data structure."""
        df = pd.DataFrame({
            "Datum": ["23-02-2026"] * 3,
            "Check-in": ["07:44", "", "19:37"],
            "Vertrek": ["Heerlen", "Heerlen", "Maastricht Randwyck"],
            "Check-uit": ["", "08:25", ""],
            "Bestemming": ["", "Maastricht Randwyck", ""],
            "Bedrag": ["", "", ""],
            "Transactie": ["Check-in", "Check-uit", "Check-in"],
            "Klasse": ["", "", ""],
            "Product": ["Student Week Vrij (2e klas)"] * 3,
            "Opmerkingen": ["", "", ""],
            "Naam": ["M. Roca Cugat"] * 3,
            "Kaartnummer": ["3528 0704 1073 7423"] * 3
        })
        
        result = detect_subscription_from_product(df, "3528 0704 1073 7423")
        assert result == "Student Week (Free weekdays, Discount weekends)"

    def test_subscription_options_are_valid(self):
        """Test that all SUBSCRIPTION_OPTIONS are strings."""
        assert len(SUBSCRIPTION_OPTIONS) == 4
        assert all(isinstance(opt, str) for opt in SUBSCRIPTION_OPTIONS)
        assert "No Subscription" in SUBSCRIPTION_OPTIONS
        assert "Student Week" in SUBSCRIPTION_OPTIONS[1]
        assert "Student Weekend" in SUBSCRIPTION_OPTIONS[2]
        assert "Other Subscription" in SUBSCRIPTION_OPTIONS

    def test_handles_nan_product_values(self):
        """Test that NaN product values are handled gracefully."""
        df = pd.DataFrame({
            "Kaartnummer": ["CARD555"] * 3,
            "Product": [pd.NA, None, "Student Week Vrij"]
        })
        
        result = detect_subscription_from_product(df, "CARD555")
        assert result == "Student Week (Free weekdays, Discount weekends)"

    def test_whitespace_in_card_numbers(self):
        """Test that whitespace in card numbers is handled correctly."""
        df = pd.DataFrame({
            "Kaartnummer": [" CARD666 ", "CARD666", "  CARD666"],
            "Product": ["Student Week Vrij"] * 3
        })
        
        result = detect_subscription_from_product(df, "CARD666")
        assert result == "Student Week (Free weekdays, Discount weekends)"

    def test_normalize_subscription_state_per_card_defaults(self):
        """Each unique card keeps its own setting; missing cards get default value."""
        kaartnummers = ["CARD1", "CARD2", "CARD3"]
        current = {
            "CARD1": "Student Week (Free weekdays, Discount weekends)",
            "CARD2": "invalid-subscription",
        }

        normalized = normalize_subscription_state(kaartnummers, current)

        assert normalized == {
            "CARD1": "Student Week (Free weekdays, Discount weekends)",
            "CARD2": "No Subscription",
            "CARD3": "No Subscription",
        }

    def test_apply_subscriptions_to_merged_trips_maps_per_card(self):
        """Merged trips receives Subscription per Kaartnummer, with defaults when missing."""
        merged_trips = pd.DataFrame({
            "Kaartnummer": ["CARD1", " CARD2 ", "CARD3"],
            "Vertrek": ["A", "B", "C"],
        })
        subscriptions = {
            "CARD1": "Student Week (Free weekdays, Discount weekends)",
            "CARD2": "Student Weekend (Discount weekdays, Free weekends)",
        }

        result = apply_subscriptions_to_merged_trips(merged_trips, subscriptions)

        assert "Subscription" in result.columns
        assert result["Subscription"].tolist() == [
            "Student Week (Free weekdays, Discount weekends)",
            "Student Weekend (Discount weekdays, Free weekends)",
            "No Subscription",
        ]

    def test_apply_subscriptions_to_merged_trips_uses_default_without_card_column(self):
        """If Kaartnummer is missing, a default Subscription column is still added."""
        merged_trips = pd.DataFrame({
            "Vertrek": ["A"],
            "Bestemming": ["B"],
        })

        result = apply_subscriptions_to_merged_trips(merged_trips, {"CARD1": SUBSCRIPTION_OPTIONS[1]})

        assert "Subscription" in result.columns
        assert result["Subscription"].tolist() == ["No Subscription"]

    def test_merge_then_apply_subscriptions_keeps_card_specific_values(self):
        """Card-specific subscription selections are written into merged trips rows."""
        trips_df = pd.DataFrame(
            [
                {
                    "Datum": "01-01-2026",
                    "Check-in": "08:00",
                    "Vertrek": "A",
                    "Check-uit": "",
                    "Bestemming": "",
                    "Bedrag": "",
                    "Transactie": "Check-in",
                    "Klasse": "2e klas",
                    "Product": "Student Week Vrij",
                    "Opmerkingen": "",
                    "Naam": "User",
                    "Kaartnummer": "CARD1",
                },
                {
                    "Datum": "01-01-2026",
                    "Check-in": "",
                    "Vertrek": "A",
                    "Check-uit": "08:30",
                    "Bestemming": "B",
                    "Bedrag": "",
                    "Transactie": "Check-uit",
                    "Klasse": "2e klas",
                    "Product": "Student Week Vrij",
                    "Opmerkingen": "",
                    "Naam": "User",
                    "Kaartnummer": "CARD1",
                },
                {
                    "Datum": "01-01-2026",
                    "Check-in": "09:00",
                    "Vertrek": "C",
                    "Check-uit": "",
                    "Bestemming": "",
                    "Bedrag": "",
                    "Transactie": "Check-in",
                    "Klasse": "2e klas",
                    "Product": "Student Weekend Vrij",
                    "Opmerkingen": "",
                    "Naam": "User",
                    "Kaartnummer": "CARD2",
                },
                {
                    "Datum": "01-01-2026",
                    "Check-in": "",
                    "Vertrek": "C",
                    "Check-uit": "09:40",
                    "Bestemming": "D",
                    "Bedrag": "",
                    "Transactie": "Check-uit",
                    "Klasse": "2e klas",
                    "Product": "Student Weekend Vrij",
                    "Opmerkingen": "",
                    "Naam": "User",
                    "Kaartnummer": "CARD2",
                },
            ]
        )
        merged = merge_check_in_out_transactions(trips_df)
        subscriptions = {
            "CARD1": "Student Week (Free weekdays, Discount weekends)",
            "CARD2": "Student Weekend (Discount weekdays, Free weekends)",
        }

        result = apply_subscriptions_to_merged_trips(merged, subscriptions)
        subscription_by_card = (
            result.groupby("Kaartnummer")["Subscription"].first().to_dict()
            if not result.empty
            else {}
        )

        assert subscription_by_card["CARD1"] == "Student Week (Free weekdays, Discount weekends)"
        assert subscription_by_card["CARD2"] == "Student Weekend (Discount weekdays, Free weekends)"
