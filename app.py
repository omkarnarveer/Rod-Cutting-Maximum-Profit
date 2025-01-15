import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the maximum profit and the optimal cuts using DP (from the provided C++ logic)
def maxprice(price, n):
    # Create a DP table to store results of subproblems
    dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    # Fill the DP table based on the logic
    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0  # Base case: no rod or no cuts available
            else:
                if j < i:
                    dp[i][j] = dp[i - 1][j]  # Can't cut a piece of length i if j < i
                else:
                    dp[i][j] = max(price[i - 1] + dp[i][j - i], dp[i - 1][j])

    # Return the maximum profit from the DP table
    return dp[n][n]

# Function to find the optimal cuts
def find_optimal_cuts(price, n):
    dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    cuts = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    # Fill the DP table based on the logic
    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
                cuts[i][j] = 0
            else:
                if j < i:
                    dp[i][j] = dp[i - 1][j]
                    cuts[i][j] = cuts[i - 1][j]
                else:
                    if price[i - 1] + dp[i][j - i] > dp[i - 1][j]:
                        dp[i][j] = price[i - 1] + dp[i][j - i]
                        cuts[i][j] = i
                    else:
                        dp[i][j] = dp[i - 1][j]
                        cuts[i][j] = cuts[i - 1][j]

    # Reconstruct the optimal cuts from the cuts table
    optimal_cuts = []
    length = n
    while length > 0:
        optimal_cuts.append(cuts[n][length])
        length -= cuts[n][length]

    return dp[n][n], optimal_cuts

# Streamlit UI for inputs
st.markdown("<h1 style='text-align: center;'>Rod Cutting Problem</h1>", unsafe_allow_html=True)

# Input Section
st.container()
with st.expander("Input Section", expanded=True):
    st.markdown("### Enter the Rod Length and Price List")
    # Input for rod length and price array
    n = st.number_input("Enter the rod length:", min_value=1, max_value=100, value=8, step=1)
    price_input = st.text_area("Enter the price list (comma-separated values for lengths 1 to n):", 
                              "1,5,8,9,10,17,17,20")
    # Convert input to a list of integers
    price = list(map(int, price_input.split(',')))

    if len(price) != n:
        st.error("Price list length must match the rod length.")

# Create the "Calculate" button
calculate_button = st.button("Calculate Maximum Profit and Visualize")

# Check if the button is clicked
if calculate_button and len(price) == n:
    # Calculate maximum profit and optimal cuts using dynamic programming
    max_profit, optimal_cuts = find_optimal_cuts(price, n)

    # Display the result
    st.subheader(f"Maximum Profit: {max_profit}")
    st.write(f"Optimal cuts (lengths): {optimal_cuts}")
    
    # Visualize the DP table and the cuts made
    dp_table = np.zeros((n+1, n+1))
    
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            dp_table[i][j] = price[j - 1] + dp_table[i - j][j] if (i - j) >= 0 else 0
    
    # Visualization Section
    st.container()
    with st.expander("Visualizations", expanded=True):
        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            # Plotting the DP table as a heatmap
            fig, ax = plt.subplots(figsize=(10, 8))  # Increased figure size
            cax = ax.matshow(dp_table, cmap='Blues')
            for i in range(n+1):
                for j in range(n+1):
                    ax.text(j, i, int(dp_table[i][j]), ha='center', va='center', color='black')

            plt.colorbar(cax)
            st.subheader("DP Table (Maximum Profit Calculation)")
            st.pyplot(fig)

        with col2:
            # Plotting the optimal cuts as a bar chart
            cut_values = [price[c-1] for c in optimal_cuts]
            cut_lengths = [f"Length {c}" for c in optimal_cuts]
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))  # Increased figure size
            ax2.bar(cut_lengths, cut_values, color='skyblue')
            ax2.set_title("Profit from Each Optimal Cut Length")
            ax2.set_xlabel("Length of Pieces")
            ax2.set_ylabel("Profit")
            
            st.subheader("Optimal Cuts Visualization")
            st.pyplot(fig2)
