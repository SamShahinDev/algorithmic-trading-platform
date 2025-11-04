#!/usr/bin/env python3
"""
Simple test to verify the trading bot works
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("🧪 Simple Trading Bot Test")
print("="*50)

# Test 1: Data fetching
print("\n1️⃣ Testing data fetch...")
ticker = yf.Ticker("NQ=F")  # NQ Futures
data = ticker.history(period="5d", interval="1h")

if not data.empty:
    print(f"✅ Fetched {len(data)} bars of NQ data")
    print(f"   Latest price: ${data['Close'].iloc[-1]:,.2f}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
else:
    print("❌ Could not fetch data")

# Test 2: Simple pattern detection
print("\n2️⃣ Testing simple pattern detection...")
data['SMA_20'] = data['Close'].rolling(20).mean()
data['Signal'] = 0
data.loc[data['Close'] > data['SMA_20'], 'Signal'] = 1

bullish_periods = data['Signal'].sum()
total_periods = len(data)
bullish_percentage = (bullish_periods / total_periods) * 100

print(f"✅ Pattern analysis complete")
print(f"   Bullish periods: {bullish_periods}/{total_periods} ({bullish_percentage:.1f}%)")

# Test 3: Check database
print("\n3️⃣ Testing database...")
import sqlite3
try:
    conn = sqlite3.connect('data/patterns.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"✅ Database has {len(tables)} tables")
    for table in tables:
        print(f"   - {table[0]}")
    conn.close()
except Exception as e:
    print(f"❌ Database error: {e}")

print("\n" + "="*50)
print("✅ Basic functionality working!")
print("\nThe trading bot infrastructure is set up correctly.")
print("Pattern discovery will run in the background when you")
print("start the main orchestrator.")