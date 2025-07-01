import matplotlib.pyplot as plt
import random
from treys import Card, Evaluator

def estimate_hand_prob(category, is_aggressive=False, in_position=False):
    base_probs = {
        "High Card": 0.15,
        "Pair": 0.35,
        "Two Pair": 0.25,
        "Three of a Kind": 0.12,
        "Straight": 0.07,
        "Flush": 0.03,
        "Full House": 0.03
    }
    prob = base_probs.get(category, 0.01)
    if is_aggressive:
        if category in ["High Card", "Pair"]:
            prob *= 0.90
        else:
            prob *= 1.10
    if in_position and category in ["Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House"]:
        prob *= 1.05
    return max(0.001, round(prob, 4))

SUITS = ['C', 'D', 'H', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
evaluator = Evaluator()

def create_deck():
    return [s + r for s in SUITS for r in RANKS]

def plot_likelihood_trace(trace):
    plt.figure(figsize=(10, 5))
    plt.plot(trace, color='steelblue')
    plt.title('MCMC Likelihood Trace')
    plt.xlabel('Iteration')
    plt.ylabel('Likelihood')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def convert_card(c):
    suit = c[0].lower()
    rank = c[1:].upper()
    return Card.new(rank + suit)

def classify_hand_category(hand, board):
    full_hand = [convert_card(c) for c in hand + board]
    score = evaluator.evaluate(full_hand[:2], full_hand[2:])
    class_rank = evaluator.get_rank_class(score)
    return evaluator.class_to_string(class_rank)

def generate_opponent_hands(known_cards, num_opponents=4):
    deck = create_deck()
    for card in known_cards:
        if card in deck:
            deck.remove(card)
    random.shuffle(deck)
    return [[deck.pop(), deck.pop()] for _ in range(num_opponents)]

def mcmc_simulation_new_likelihood(my_hand, board, iterations=1000, burn_in=100):
    known_cards = my_hand + board
    trace = []
    accepted = 0
    sampled_states = []

    opp_hands = generate_opponent_hands(known_cards)
    current_likelihood = 1.0
    current_state = []

    for i, hand in enumerate(opp_hands):
        cat = classify_hand_category(hand, board)
        aggr = random.randint(0, 1)
        pos = int(i == 3)
        prob = estimate_hand_prob(cat, aggr, pos)
        current_likelihood *= prob
        current_state.append({
            "hand": hand,
            "category": cat,
            "aggression": aggr,
            "position": pos,
            "prob": prob
        })

    for step in range(iterations):
        proposed_hands = generate_opponent_hands(known_cards)
        proposed_state = []
        prop_likelihood = 1.0

        for i, hand in enumerate(proposed_hands):
            cat = classify_hand_category(hand, board)
            aggr = random.randint(0, 1)
            pos = int(i == 3)
            prob = estimate_hand_prob(cat, aggr, pos)
            prop_likelihood *= prob
            proposed_state.append({
                "hand": hand,
                "category": cat,
                "aggression": aggr,
                "position": pos,
                "prob": prob
            })

        accept_prob = min(1.0, prop_likelihood / current_likelihood)
        if random.random() < accept_prob:
            current_state = proposed_state
            current_likelihood = prop_likelihood
            accepted += 1

        if step >= burn_in:
            trace.append(current_likelihood)
            sampled_states.append(current_state)

    return trace, accepted / iterations, sampled_states

def print_sampled_state(state, my_hand, board, index=0):
    print(f"\n=== Sampled MCMC Table State #{index + 1} ===")
    print(f"Your hand: {my_hand}")
    print(f"Board: {board}\n")
    for i, opp in enumerate(state):
        print(f"Opponent {i+1}: {opp['hand']} | {opp['category']} | P: {opp['prob']:.3f} | Aggression: {opp['aggression']} | Position: {opp['position']}")

from collections import defaultdict

def compute_category_distributions(sampled_states):
    """
    Given the list of sampled states, computes the frequency distribution
    of hand categories for each opponent.
    """
    category_counts = [defaultdict(int) for _ in range(4)]  # one dict per opponent

    for state in sampled_states:
        for i, opp in enumerate(state):
            category_counts[i][opp['category']] += 1

    # Normalize to get probabilities
    category_distributions = []
    total_samples = len(sampled_states)
    for i in range(4):
        dist = {
            cat: round(count / total_samples, 4)
            for cat, count in category_counts[i].items()
        }
        category_distributions.append(dist)

    return category_distributions

def print_category_distributions(distributions):
    """
    Pretty-prints the hand category probabilities for each opponent.
    """
    for i, dist in enumerate(distributions):
        print(f"\n=== Opponent {i + 1} Hand Category Probabilities ===")
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        for cat, prob in sorted_dist:
            print(f"{cat:<20}: {prob:.2%}")
    print("\nNote: Probabilities are estimated from sampled MCMC states.")

def plot_hand_category_distributions(distributions):
    """
    Generate bar plots of hand category probabilities for each opponent.
    """
    num_opponents = len(distributions)
    fig, axs = plt.subplots(1, num_opponents, figsize=(5 * num_opponents, 5), sharey=True)
    if num_opponents == 1:
        axs = [axs]

    for i, dist in enumerate(distributions):
        categories = list(dist.keys())
        probabilities = list(dist.values())
        axs[i].bar(range(len(categories)), probabilities)
        axs[i].set_title(f"Opponent {i+1}")
        axs[i].set_ylabel("Probability")
        axs[i].set_xticks(range(len(categories)))
        axs[i].set_xticklabels(categories, rotation=45, ha='right')
        axs[i].set_ylim(0, 1)


    plt.suptitle("Hand Category Probabilities Per Opponent")
    plt.tight_layout()
    plt.show()

from treys import Evaluator, Card

def estimate_winrate_from_samples(my_hand, board, sampled_states):
    evaluator = Evaluator()
    my_cards = [Card.new(c[1] + c[0].lower()) for c in my_hand]
    board_cards = [Card.new(c[1] + c[0].lower()) for c in board]

    win, tie, loss = 0, 0, 0

    for state in sampled_states:
        my_score = evaluator.evaluate(my_cards, board_cards)
        best_score = my_score
        result = "win"

        for opp in state:
            opp_cards = [Card.new(c[1] + c[0].lower()) for c in opp['hand']]
            opp_score = evaluator.evaluate(opp_cards, board_cards)
            if opp_score < best_score:
                result = "loss"
                break
            elif opp_score == best_score:
                result = "tie"
        
        if result == "win":
            win += 1
        elif result == "tie":
            tie += 1
        else:
            loss += 1

    total = win + tie + loss
    return {
        "winrate": round(win / total, 4),
        "tie_rate": round(tie / total, 4),
        "loss_rate": round(loss / total, 4),
        "total_samples": total
    }

def print_sampled_state_detailed(state, my_hand, board, index=0):
    from operator import itemgetter
    print(f"\n=== Sampled MCMC Table State #{index + 1} ===")
    print(f"Your hand: {my_hand}")
    print(f"Board: {board}")
    
    # Classify player's hand
    my_category = classify_hand_category(my_hand, board)
    print(f"Your category: {my_category}\n")
    
    # Sort opponents by descending probability
    sorted_state = sorted(state, key=itemgetter('prob'), reverse=True)
    
    total_likelihood = 1.0
    for opp in sorted_state:
        total_likelihood *= opp['prob']

    for i, opp in enumerate(sorted_state):
        print(f"Opponent {i+1}: {opp['hand']} | {opp['category']} | "
              f"P: {opp['prob']:.3f} | Aggression: {opp['aggression']} | "
              f"Position: {opp['position']}")
    
    print(f"\nTotal Likelihood of this state: {total_likelihood:.5f}")

def plot_winrate_pie(winrate_result):
    """
    Plots a pie chart for win, tie, and loss rates.
    """
    labels = ['Win', 'Tie', 'Loss']
    sizes = [
        winrate_result['winrate'],
        winrate_result['tie_rate'],
        winrate_result['loss_rate']
    ]
    colors = ['#4CAF50', '#FFEB3B', '#F44336']  # green, yellow, red

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Estimated Winrate Breakdown')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Example usage
if __name__ == "__main__":
    my_hand = ['C9', 'H9']
    board = ['SK', 'S7', 'H6', 'H5']
    trace, acc_rate, samples = mcmc_simulation_new_likelihood(my_hand, board)
    plot_likelihood_trace(trace)
    distributions = compute_category_distributions(samples)
    print_category_distributions(distributions)
    plot_hand_category_distributions(distributions)
    result = estimate_winrate_from_samples(my_hand, board, samples)
    plot_winrate_pie(result)


    print(f"My Estimated Winrate: {result['winrate']:.2%}")
    print(f"Tie Rate: {result['tie_rate']:.2%}")
    print(f"Loss Rate: {result['loss_rate']:.2%}")

    print(f"Trace length: {len(trace)}, Acceptance Rate: {acc_rate:.2%}")
    if samples:
        print_sampled_state_detailed(samples[0], my_hand, board)