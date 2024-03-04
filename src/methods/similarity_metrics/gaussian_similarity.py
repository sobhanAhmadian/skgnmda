import numpy as np


def generate_dict_id(pairs_array):
    # pairs_array[:,:1]:disease, pairs_array[:,1:2]:microbe, pairs_array[:,2:3]:label
    first_term = set()
    second_term = set()
    for i in range(len(pairs_array)):
        if pairs_array[i, 2] == 1:
            first_term.add(pairs_array[i, 0])
            second_term.add(pairs_array[i, 1])
    first_term2id = {}
    first_id = 0
    for term in first_term:
        first_term2id[term] = first_id
        first_id += 1
    second_term2id = {}
    second_id = 0
    for term in second_term:
        second_term2id[term] = second_id
        second_id += 1
    return first_term2id, second_term2id


def generate_interaction_profile(pairs_array):
    first_term2id, second_term2id = generate_dict_id(pairs_array)
    print("first_term2id=", first_term2id, "second_term2id=", second_term2id)
    interaction_profile = np.zeros((len(first_term2id), len(second_term2id)))
    for i in range(len(pairs_array)):
        if pairs_array[i, 2] == 1:
            interaction_profile[
                first_term2id[pairs_array[i, 0]], second_term2id[pairs_array[i, 1]]
            ] = 1
    return interaction_profile, first_term2id, second_term2id


def gaussian_similarity(interaction_profile):
    nd = len(interaction_profile)
    nm = len(interaction_profile[0])

    # generate disease similarity
    gaussian_d = np.zeros((nd, nd))
    gama_d = nd / (pow(np.linalg.norm(interaction_profile), 2))
    d_matrix = np.dot(interaction_profile, interaction_profile.T)
    for i in range(nd):
        j = i
        while j < nd:
            gaussian_d[i, j] = np.exp(
                -gama_d * (d_matrix[i, i] + d_matrix[j, j] - 2 * d_matrix[i, j])
            )
            j += 1
    gaussian_d = gaussian_d + gaussian_d.T - np.diag(np.diag(gaussian_d))

    # generate microbe similarity
    gaussian_m = np.zeros((nm, nm))
    gama_m = nm / (pow(np.linalg.norm(interaction_profile), 2))
    m_matrix = np.dot(interaction_profile.T, interaction_profile)
    for l in range(nm):
        k = l
        while k < nm:
            gaussian_m[l, k] = np.exp(
                -gama_m * (m_matrix[l, l] + m_matrix[k, k] - 2 * m_matrix[l, k])
            )
            k += 1
    gaussian_m = gaussian_m + gaussian_m.T - np.diag(np.diag(gaussian_m))
    return gaussian_d, gaussian_m


def get_gaussian_similarity(all_data, test_data):
    """
    all_data : numpy array in shape of n * 3 - [entity - entity - is associated]
    data     : numpy array in shape of n * 3 - [entity - entity - is associated]
    """
    (
        interaction_profile,
        disease_term2id,
        microbe_term2id,
    ) = generate_interaction_profile(np.array(all_data))
    for i in range(len(test_data)):
        if test_data[i, 2] == 1:
            interaction_profile[
                disease_term2id[test_data[i, 0]], microbe_term2id[test_data[i, 1]]
            ] = 0

    gaussian_d, gaussian_m = gaussian_similarity(interaction_profile)

    disease_similarities = np.zeros((all_data.shape[0], gaussian_d.shape[0]))
    microbe_similarities = np.zeros((all_data.shape[0], gaussian_m.shape[0]))
    for i in range(all_data.shape[0]):
        disease_similarities[i, :] = gaussian_d[disease_term2id[all_data[i, 0]], :]
        microbe_similarities[i, :] = gaussian_m[microbe_term2id[all_data[i, 1]], :]

    test_indices = []
    for j in range(test_data.shape[0]):
        for i in range(all_data.shape[0]):
            if (all_data[i, :] == test_data[j, :]).all():
                test_indices.append(i)
                break

    train_disease_similarities = np.delete(disease_similarities, test_indices, 0)
    train_microbe_similarities = np.delete(microbe_similarities, test_indices, 0)
    test_disease_similarities = disease_similarities[test_indices, :]
    test_microbe_similarities = microbe_similarities[test_indices, :]
    return (
        train_disease_similarities,
        train_microbe_similarities,
        test_disease_similarities,
        test_microbe_similarities,
    )
