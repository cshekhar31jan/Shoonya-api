import pandas as pd
import numpy as np
import os

def generate_dummy_data(filename="stock_data_5min.csv"):
    """
    Generates a dummy CSV file if it doesn't exist, for demonstration purposes.
    """
    if os.path.exists(filename):
        print(f"'{filename}' already exists. Using existing file.")
        return

    print(f"'{filename}' not found. Generating a dummy file for demonstration...")
    # Create a time range for 2 days of 5-minute data
    timestamps = pd.date_range(start="2023-10-26 09:15:00", end="2023-10-27 15:30:00", freq="5min")
    n = len(timestamps)
    
    # Start with a base price
    price = 100
    opens = []
    highs = []
    lows = []
    closes = []

    # Generate some random walk data
    for _ in range(n):
        opens.append(price)
        high = price + np.random.uniform(0, 0.5)
        low = price - np.random.uniform(0, 0.5)
        close = np.random.uniform(low, high)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        # Next candle's open is this candle's close
        price = close

    data = {
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    }
    df = pd.DataFrame(data)

    # Calculate the percentage change for each candle
    df['%_change'] = ((df['close'] - df['open']) / df['open']) * 100
    
    # Introduce some spikes to trigger the strategy
    spike_indices = np.random.choice(df.index, 10, replace=False)
    df.loc[spike_indices, '%_change'] = np.random.uniform(0.085, 0.15, 10)
    
    df.to_csv(filename, index=False)
    print(f"Dummy file '{filename}' created successfully.")


def run_trading_simulation(df):
    """
    Runs the backtesting simulation based on the specified trading logic.

    Args:
        df (pd.DataFrame): DataFrame with stock data, including 'open', 'high', 'low',
                           'close', and '%_change' columns.

    Returns:
        tuple: A tuple containing total profit/loss and a DataFrame of all trades.
    """
    print("\n--- Starting Trading Simulation ---")
    
    # State variables
    in_position = False
    buy_price = 0
    stop_loss_price = 0
    total_profit_loss = 0.0
    trades = []

    # We iterate up to the second to last row because we need the 'next' candle's open price
    for i in range(len(df) - 1):
        current_candle = df.iloc[i]
        next_candle = df.iloc[i+1]

        # --- LOGIC WHEN WE ARE HOLDING A POSITION ---
        if in_position:
            # 1. CHECK FOR STOP-LOSS FIRST (highest priority exit)
            # The stop-loss is checked against the low of the *next* candle.
            if next_candle['low'] <= stop_loss_price:
                sell_price = stop_loss_price  # Assume stop-loss executes at the set price
                profit = sell_price - buy_price
                total_profit_loss += profit
                
                trades.append({
                    'Entry_Timestamp': entry_timestamp,
                    'Buy_Price': buy_price,
                    'Exit_Timestamp': next_candle['timestamp'],
                    'Sell_Price': sell_price,
                    'Profit/Loss': profit,
                    'Exit_Reason': 'Stop-Loss'
                })
                
                # Exit position
                in_position = False
                print(f"SELL (Stop-Loss) at {sell_price:.2f} on {next_candle['timestamp']}")
                continue # Move to the next candle

            # 2. CHECK FOR NORMAL EXIT CONDITION
            # Exit at the opening of the next candle if the *current* candle's %change is < 0.085
            if current_candle['%_change'] < 0.085:
                sell_price = next_candle['open']
                profit = sell_price - buy_price
                total_profit_loss += profit

                trades.append({
                    'Entry_Timestamp': entry_timestamp,
                    'Buy_Price': buy_price,
                    'Exit_Timestamp': next_candle['timestamp'],
                    'Sell_Price': sell_price,
                    'Profit/Loss': profit,
                    'Exit_Reason': 'Signal'
                })
                
                # Exit position
                in_position = False
                print(f"SELL (Signal) at {sell_price:.2f} on {next_candle['timestamp']}")
        
        # --- LOGIC WHEN WE ARE NOT HOLDING A POSITION ---
        else: # if not in_position
            # CHECK FOR ENTRY CONDITION
            # Buy at the opening of the next candle if the *current* candle's %change is >= 0.085
            if current_candle['%_change'] >= 0.085:
                # Enter position
                in_position = True
                buy_price = next_candle['open']
                entry_timestamp = next_candle['timestamp']
                
                # Calculate Stop-Loss
                # (buy price - ((high - low)/2 of the prior candle))
                # The "prior candle" is the one that triggered the buy signal (i.e., current_candle)
                prior_candle_range = (current_candle['high'] - current_candle['low']) / 2
                stop_loss_price = buy_price - prior_candle_range
                
                print(f"\nBUY signal on {current_candle['timestamp']}")
                print(f"  -> Executing BUY at {buy_price:.2f} on {entry_timestamp}")
                print(f"  -> Setting Stop-Loss at {stop_loss_price:.2f}")

    # If the simulation ends while still in a position, we can choose to close it
    # at the last available price (last candle's close)
    if in_position:
        last_candle = df.iloc[-1]
        sell_price = last_candle['close']
        profit = sell_price - buy_price
        total_profit_loss += profit
        
        trades.append({
            'Entry_Timestamp': entry_timestamp,
            'Buy_Price': buy_price,
            'Exit_Timestamp': last_candle['timestamp'],
            'Sell_Price': sell_price,
            'Profit/Loss': profit,
            'Exit_Reason': 'End of Data'
        })
        print(f"\nSimulation ended. Closing final position at {sell_price:.2f}.")

    print("\n--- Simulation Complete ---")
    return total_profit_loss, pd.DataFrame(trades)

# --- Main execution block ---
if __name__ == "__main__":
    CSV_FILE = "backend\nifty_5min_data.csv"

    # Step 1: Generate dummy data if the file doesn't exist
    # generate_dummy_data(CSV_FILE)

    # Step 2: Load the data from the CSV file
    try:
        # We need to make sure the column name for % change is consistent.
        # Let's read the CSV and then standardize the column name.
        df = pd.read_csv(CSV_FILE)
        pd.show()
        
        # # Standardize column name: find a column with '%' or 'change' and rename it
        # for col in df.columns:
        #     if '%' in col or 'change' in col.lower():
        #         df.rename(columns={col: '%_change'}, inplace=True)
        #         print(f"Renamed column '{col}' to '%_change'")
        #         break

        # if '%_change' not in df.columns:
        #     raise ValueError("'% change' column not found in the CSV. Please ensure it exists.")

    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        exit()
        
    # Step 3: Prepare the DataFrame
    # Ensure timestamp is a datetime object and sort the data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\nData loaded and prepared successfully.")
    print("DataFrame head:")
    print(df.head())

    # Step 4: Run the simulation
    final_pnl, trades_df = run_trading_simulation(df)

    # Step 5: Display the results
    print("\n" + "="*50)
    print(" " * 15 + "SIMULATION RESULTS")
    print("="*50)
    print(f"Final Total Profit/Loss: {final_pnl:.2f}")
    
    if not trades_df.empty:
        print(f"Total Trades Made: {len(trades_df)}")
        print("\n--- Detailed Trade Log ---")
        print(trades_df.to_string())
    else:
        print("No trades were executed based on the strategy.")
    print("="*50)