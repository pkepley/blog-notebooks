import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import binom

def simulate_giftcards(n_drinks_on_card = 50, n_cards = 2):
    drinks_left = np.array([n_drinks_on_card for i in range(n_cards)], dtype='int')

    while True:
        card_used = np.random.randint(0, n_cards)
        if drinks_left[card_used] == 0:
            break
        else:
            drinks_left[card_used] = drinks_left[card_used] - 1

    return drinks_left

def simulate_process(n_runs = 10000, n_drinks_on_card = 50, n_cards = 2):
    n_drinks_left = np.zeros(n_runs, dtype=int)

    for i in range(n_runs):
        n_drinks_left[i] = simulate_giftcards(n_drinks_on_card, n_cards).sum()

    return n_drinks_left

def plot_riddler_results(n_runs = 10000, n_drinks_on_card = 50):
    # This is fixed
    n_cards = 2
    
    # Simulate results / print info
    n_drinks_left_sim = simulate_process(n_runs)
    avg_remaining_sim = np.mean(n_drinks_left_sim)
    prob_nonzero_sim  = np.count_nonzero(n_drinks_left_sim) / n_runs
    print("Simulation results")
    print("-------------------------------")
    print("Average drinks remaining : ", avg_remaining_sim)
    print("Probability drinks remain: ", prob_nonzero_sim)
    print("")

    # Solve analytically
    n_left = np.array(range(0, n_drinks_on_card+1))
    p_left = binom.pmf(n_drinks_on_card, 2 * n_drinks_on_card - n_left, 0.5)
    avg_remaining_analytic = np.sum(n_left * p_left)
    prob_nonzero_analytic = 1.0 - p_left[0]
    print("Analytic results")
    print("-------------------------------")
    print("Average drinks remaining : ", avg_remaining_analytic)
    print("Probability drinks remain: ", prob_nonzero_analytic)
    print("") 

    # Plot the figure with comparison
    fig, ax = plt.subplots()
    ax.hist(n_drinks_left_sim, bins=range(0, n_drinks_on_card+1),
            density=True, label="Simulated")
    ax.plot(n_left, p_left, drawstyle="steps-post",
            label = "Analytic")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    plt.show()

if __name__ == "__main__":
    np.random.seed(8675309)
    style.use('fivethirtyeight')
    plot_riddler_results(100000)
