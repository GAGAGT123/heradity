import csv
import itertools
import sys

# Probabilities for genes, traits, and genetic mutations
# Represents prior probabilities for genetic and trait distributions
PROBS = {
    # Probability of number of gene copies in an individual (0, 1, or 2 copies)
    "gene": {
        2: 0.01,  # 1% probability of having two gene copies
        1: 0.03,  # 3% probability of having one gene copy
        0: 0.96   # 96% probability of having no gene copies
    },
    
    # Probability of trait expression based on gene copy count
    "trait": {
        # With two gene copies
        2: {True: 0.65, False: 0.35},  # 65% probability of trait expression
        # With one gene copy
        1: {True: 0.56, False: 0.44},  # 56% probability of trait expression
        # With no gene copies
        0: {True: 0.01, False: 0.99}   # 1% probability of trait expression
    },
    
    # Mutation rate: probability of gene transmission being different than expected
    "mutation": 0.01
}


def main():
    """
    Main function that controls program flow.
    Calculates probability distributions for genes and traits across individuals.
    """
    
    # Validate command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    
    # Load data from CSV file
    people = load_data(sys.argv[1])
    
    # Initialize data structure to store probabilities
    # For each person, we store probabilities for genes (0, 1, 2) and traits (True/False)
    probabilities = {
        person: {
            "gene": {2: 0, 1: 0, 0: 0},  # Accumulating probabilities for gene counts
            "trait": {True: 0, False: 0}  # Accumulating probabilities for traits
        }
        for person in people
    }
    
    # Extract names of all individuals
    names = set(people)
    
    # Full enumeration of all possible gene and trait configurations
    # Iterate through all possible subsets of individuals who have the trait
    for have_trait in powerset(names):
        # Verify this configuration matches the evidence from the data
        fails_evidence = False
        for person in names:
            # If trait is known in the original data
            if people[person]["trait"] is not None:
                # Check if this configuration contradicts the evidence
                if people[person]["trait"] != (person in have_trait):
                    fails_evidence = True
                    break
        
        # Skip this configuration if it contradicts the evidence
        if fails_evidence:
            continue
        
        # Enumerate all possible gene configurations
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Calculate joint probability for this specific configuration
                p = joint_probability(people, one_gene, two_genes, have_trait)
                # Update probability tables with this configuration's contribution
                update(probabilities, one_gene, two_genes, have_trait, p)
    
    # Normalize probabilities so they sum to 1
    normalize(probabilities)
    
    # Print final probability distributions for each individual
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                print(f"    {value}: {probabilities[person][field][value]:.4f}")


def load_data(filename):
    """
    Load family data from a CSV file.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        dict: Dictionary where keys are person names and values are their data
              including mother, father, and trait information.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            # Convert trait values: "1" -> True, "0" -> False, empty -> None
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,   # Empty string becomes None
                "father": row["father"] or None,   # Empty string becomes None
                "trait": (
                    True if row["trait"] == "1" else
                    False if row["trait"] == "0" else
                    None  # Unknown trait status
                )
            }
    return data


def powerset(s):
    """
    Generate all possible subsets of a set.
    
    Args:
        s (set): Input set
        
    Returns:
        list: List of all subsets as sets
    """
    s = list(s)
    return [
        set(combo) for combo in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Calculate joint probability of a specific gene and trait configuration.
    
    Args:
        people (dict): Dictionary of people and their data
        one_gene (set): Set of people with one gene copy
        two_genes (set): Set of people with two gene copies
        have_trait (set): Set of people expressing the trait
        
    Returns:
        float: Joint probability of the entire configuration
    """
    probability = 1  # Initialize joint probability
    
    for person in people:
        mother = people[person]["mother"]
        father = people[person]["father"]
        
        # Determine gene count for this person based on configuration
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0
        
        # Determine if person has trait in this configuration
        trait = person in have_trait
        
        # Calculate probability of gene inheritance
        if mother is None and father is None:
            # No parents: use prior probability from PROBS
            gene_prob = PROBS["gene"][genes]
        else:
            # Has parents: calculate inheritance probability
            def pass_gene(parent):
                """
                Calculate probability that a parent passes the gene to child.
                
                Args:
                    parent (str): Parent's name
                    
                Returns:
                    float: Probability of gene transmission
                """
                if parent in two_genes:
                    # Parent has 2 copies: high probability minus mutation chance
                    return 1 - PROBS["mutation"]
                elif parent in one_gene:
                    # Parent has 1 copy: 50% chance
                    return 0.5
                else:
                    # Parent has 0 copies: only possible via mutation
                    return PROBS["mutation"]
            
            # Get transmission probabilities from mother and father
            m = pass_gene(mother)
            f = pass_gene(father)
            
            # Calculate gene inheritance probability based on child's gene count
            if genes == 2:
                # Child gets gene from both parents
                gene_prob = m * f
            elif genes == 1:
                # Child gets gene from one parent but not the other
                gene_prob = m * (1 - f) + (1 - m) * f
            else:
                # Child gets gene from neither parent
                gene_prob = (1 - m) * (1 - f)
        
        # Get trait probability based on gene count
        trait_prob = PROBS["trait"][genes][trait]
        
        # Multiply into joint probability (assuming independence)
        probability *= gene_prob * trait_prob
    
    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Update probability tables with a configuration's contribution.
    
    Args:
        probabilities (dict): Probability tables to update
        one_gene (set): People with one gene copy in this configuration
        two_genes (set): People with two gene copies in this configuration
        have_trait (set): People with trait in this configuration
        p (float): Probability of this configuration
    """
    for person in probabilities:
        # Update gene probability distributions
        if person in two_genes:
            probabilities[person]["gene"][2] += p
        elif person in one_gene:
            probabilities[person]["gene"][1] += p
        else:
            probabilities[person]["gene"][0] += p
        
        # Update trait probability distributions
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Normalize probabilities so each distribution sums to 1.
    
    Args:
        probabilities (dict): Probability tables to normalize
    """
    for person in probabilities:
        # Normalize gene probabilities
        gene_sum = sum(probabilities[person]["gene"].values())
        for g in probabilities[person]["gene"]:
            probabilities[person]["gene"][g] /= gene_sum
        
        # Normalize trait probabilities
        trait_sum = sum(probabilities[person]["trait"].values())
        for t in probabilities[person]["trait"]:
            probabilities[person]["trait"][t] /= trait_sum


if __name__ == "__main__":
    main()

