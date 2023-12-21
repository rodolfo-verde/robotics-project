import numpy as np
import matplotlib.pyplot as plt

# combine all set_complete_test data into one file, including augmentated data
# class_names = ['a', 'b', 'c', '1', '2', '3', 'stopp', 'rex', 'other']

# From Train_Data/
# class a, augmentated
a_1 = np.load('audio_processing\Train_Data\set_complete_test_a_mfcc_cutout.npy')
a_2 = np.load('audio_processing\Train_Data\set_complete_test_a_mfcc_mixed_frequency_masking.npy')
a_3 = np.load('audio_processing\Train_Data\set_complete_test_a_mfcc_mixup.npy')
a_4 = np.load('audio_processing\Train_Data\set_complete_test_a_mfcc_sample_pairing.npy')
a_5 = np.load('audio_processing\Train_Data\set_complete_test_a_mfcc_specaugment.npy')
a_6 = np.load('audio_processing\Train_Data\set_complete_test_a_mfcc_specmix.npy')
a_7 = np.load('audio_processing\Train_Data\set_complete_test_a_mfcc_vh_mixup.npy')

# load a labels
a_label = np.load('audio_processing\Train_Data\set_a_200_label.npy')  

# combine all a data
a_mfcc = np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7), axis=0)

# create a new label array the same size as a_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
a_label_large = np.tile(a_label, (a_mfcc.shape[0] // a_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
a_label_new = a_label_large[:a_mfcc.shape[0], :]

# print shapes
print('Shape of a_mfcc = ', a_mfcc.shape)
print('Shape of a_label = ', a_label_new.shape)

# class b, augmentated
b_1 = np.load('audio_processing\Train_Data\set_complete_test_b_mfcc_cutout.npy')
b_2 = np.load('audio_processing\Train_Data\set_complete_test_b_mfcc_mixed_frequency_masking.npy')
b_3 = np.load('audio_processing\Train_Data\set_complete_test_b_mfcc_mixup.npy')
b_4 = np.load('audio_processing\Train_Data\set_complete_test_b_mfcc_sample_pairing.npy')
b_5 = np.load('audio_processing\Train_Data\set_complete_test_b_mfcc_specaugment.npy')
b_6 = np.load('audio_processing\Train_Data\set_complete_test_b_mfcc_specmix.npy')
b_7 = np.load('audio_processing\Train_Data\set_complete_test_b_mfcc_vh_mixup.npy')

# load b labels
b_label = np.load('audio_processing\Train_Data\set_b_200_label.npy')

# combine all b data
b_mfcc = np.concatenate((b_1, b_2, b_3, b_4, b_5, b_6, b_7), axis=0)

# create a new label array the same size as b_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
b_label_large = np.tile(b_label, (b_mfcc.shape[0] // b_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
b_label_new = b_label_large[:b_mfcc.shape[0], :]
# print shapes
print('Shape of b_mfcc = ', b_mfcc.shape)
print('Shape of b_label = ', b_label_new.shape)

# class c, augmentated
c_1 = np.load('audio_processing\Train_Data\set_complete_test_c_mfcc_cutout.npy')
c_2 = np.load('audio_processing\Train_Data\set_complete_test_c_mfcc_mixed_frequency_masking.npy')
c_3 = np.load('audio_processing\Train_Data\set_complete_test_c_mfcc_mixup.npy')
c_4 = np.load('audio_processing\Train_Data\set_complete_test_c_mfcc_sample_pairing.npy')
c_5 = np.load('audio_processing\Train_Data\set_complete_test_c_mfcc_specaugment.npy')
c_6 = np.load('audio_processing\Train_Data\set_complete_test_c_mfcc_specmix.npy')
c_7 = np.load('audio_processing\Train_Data\set_complete_test_c_mfcc_vh_mixup.npy')

# load c labels
c_label = np.load('audio_processing\Train_Data\set_c_200_label.npy')

# combine all c data
c_mfcc = np.concatenate((c_1, c_2, c_3, c_4, c_5, c_6, c_7), axis=0)

# create a new label array the same size as c_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
c_label_large = np.tile(c_label, (c_mfcc.shape[0] // c_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
c_label_new = c_label_large[:c_mfcc.shape[0], :]
# print shapes
print('Shape of c_mfcc = ', c_mfcc.shape)
print('Shape of c_label = ', c_label_new.shape)

# class 1, augmentated
one_1 = np.load('audio_processing\Train_Data\set_complete_test_1_mfcc_cutout.npy')
one_2 = np.load('audio_processing\Train_Data\set_complete_test_1_mfcc_mixed_frequency_masking.npy')
one_3 = np.load('audio_processing\Train_Data\set_complete_test_1_mfcc_mixup.npy')
one_4 = np.load('audio_processing\Train_Data\set_complete_test_1_mfcc_sample_pairing.npy')
one_5 = np.load('audio_processing\Train_Data\set_complete_test_1_mfcc_specaugment.npy')
one_6 = np.load('audio_processing\Train_Data\set_complete_test_1_mfcc_specmix.npy')
one_7 = np.load('audio_processing\Train_Data\set_complete_test_1_mfcc_vh_mixup.npy')

# class 1 labels
one_label = np.load('audio_processing\Train_Data\set_eins_200_label.npy')

# combine all 1 data
one_mfcc = np.concatenate((one_1, one_2, one_3, one_4, one_5, one_6, one_7), axis=0)

# create a new label array the same size as one_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
one_label_large = np.tile(one_label, (one_mfcc.shape[0] // one_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
one_label_new = one_label_large[:one_mfcc.shape[0], :]
# print shapes
print('Shape of one_mfcc = ', one_mfcc.shape)
print('Shape of one_label = ', one_label_new.shape)

# class 2, augmentated
two_1 = np.load('audio_processing\Train_Data\set_complete_test_2_mfcc_cutout.npy')
two_2 = np.load('audio_processing\Train_Data\set_complete_test_2_mfcc_mixed_frequency_masking.npy')
two_3 = np.load('audio_processing\Train_Data\set_complete_test_2_mfcc_mixup.npy')
two_4 = np.load('audio_processing\Train_Data\set_complete_test_2_mfcc_sample_pairing.npy')
two_5 = np.load('audio_processing\Train_Data\set_complete_test_2_mfcc_specaugment.npy')
two_6 = np.load('audio_processing\Train_Data\set_complete_test_2_mfcc_specmix.npy')
two_7 = np.load('audio_processing\Train_Data\set_complete_test_2_mfcc_vh_mixup.npy')

# class 2 labels
two_label = np.load('audio_processing\Train_Data\set_zwei_200_label.npy')

# combine all 2 data
two_mfcc = np.concatenate((two_1, two_2, two_3, two_4, two_5, two_6, two_7), axis=0)

# create a new label array the same size as two_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
two_label_large = np.tile(two_label, (two_mfcc.shape[0] // two_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
two_label_new = two_label_large[:two_mfcc.shape[0], :]
# print shapes
print('Shape of two_mfcc = ', two_mfcc.shape)
print('Shape of two_label = ', two_label_new.shape)

# class 3, augmentated
three_1 = np.load('audio_processing\Train_Data\set_complete_test_3_mfcc_cutout.npy')
three_2 = np.load('audio_processing\Train_Data\set_complete_test_3_mfcc_mixed_frequency_masking.npy')
three_3 = np.load('audio_processing\Train_Data\set_complete_test_3_mfcc_mixup.npy')
three_4 = np.load('audio_processing\Train_Data\set_complete_test_3_mfcc_sample_pairing.npy')
three_5 = np.load('audio_processing\Train_Data\set_complete_test_3_mfcc_specaugment.npy')
three_6 = np.load('audio_processing\Train_Data\set_complete_test_3_mfcc_specmix.npy')
three_7 = np.load('audio_processing\Train_Data\set_complete_test_3_mfcc_vh_mixup.npy')

# class 3 labels
three_label = np.load('audio_processing\Train_Data\set_drei_200_label.npy')

# combine all 3 data
three_mfcc = np.concatenate((three_1, three_2, three_3, three_4, three_5, three_6, three_7), axis=0)

# create a new label array the same size as three_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
three_label_large = np.tile(three_label, (three_mfcc.shape[0] // three_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
three_label_new = three_label_large[:three_mfcc.shape[0], :]
# print shapes
print('Shape of three_mfcc = ', three_mfcc.shape)
print('Shape of three_label = ', three_label_new.shape)

# class stopp, augmentated
stopp_1 = np.load('audio_processing\Train_Data\set_complete_test_stopp_mfcc_cutout.npy')
stopp_2 = np.load('audio_processing\Train_Data\set_complete_test_stopp_mfcc_mixed_frequency_masking.npy')
stopp_3 = np.load('audio_processing\Train_Data\set_complete_test_stopp_mfcc_mixup.npy')
stopp_4 = np.load('audio_processing\Train_Data\set_complete_test_stopp_mfcc_sample_pairing.npy')
stopp_5 = np.load('audio_processing\Train_Data\set_complete_test_stopp_mfcc_specaugment.npy')
stopp_6 = np.load('audio_processing\Train_Data\set_complete_test_stopp_mfcc_specmix.npy')
stopp_7 = np.load('audio_processing\Train_Data\set_complete_test_stopp_mfcc_vh_mixup.npy')

# class stopp labels
stopp_label = np.load('audio_processing\Train_Data\set_stopp_200_label.npy')

# combine all stopp data
stopp_mfcc = np.concatenate((stopp_1, stopp_2, stopp_3, stopp_4, stopp_5, stopp_6, stopp_7), axis=0)

# create a new label array the same size as stopp_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
stopp_label_large = np.tile(stopp_label, (stopp_mfcc.shape[0] // stopp_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
stopp_label_new = stopp_label_large[:stopp_mfcc.shape[0], :]
# print shapes
print('Shape of stopp_mfcc = ', stopp_mfcc.shape)
print('Shape of stopp_label = ', stopp_label_new.shape)

# class rex, augmentated
rex_1 = np.load('audio_processing\Train_Data\set_complete_test_rex_mfcc_cutout.npy')
rex_2 = np.load('audio_processing\Train_Data\set_complete_test_rex_mfcc_mixed_frequency_masking.npy')
rex_3 = np.load('audio_processing\Train_Data\set_complete_test_rex_mfcc_mixup.npy')
rex_4 = np.load('audio_processing\Train_Data\set_complete_test_rex_mfcc_sample_pairing.npy')
rex_5 = np.load('audio_processing\Train_Data\set_complete_test_rex_mfcc_specaugment.npy')
rex_6 = np.load('audio_processing\Train_Data\set_complete_test_rex_mfcc_specmix.npy')
rex_7 = np.load('audio_processing\Train_Data\set_complete_test_rex_mfcc_vh_mixup.npy')

# class rex labels
rex_label = np.load('audio_processing\Train_Data\set_rex_200_label.npy')

# combine all rex data
rex_mfcc = np.concatenate((rex_1, rex_2, rex_3, rex_4, rex_5, rex_6, rex_7), axis=0)

# create a new label array the same size as rex_mfcc.shape[0]
# Tile labels to a size larger than MFCC data
rex_label_large = np.tile(rex_label, (rex_mfcc.shape[0] // rex_label.shape[0] + 1, 1))

# Slice labels to match the exact size of MFCC data
rex_label_new = rex_label_large[:rex_mfcc.shape[0], :]
# print shapes
print('Shape of rex_mfcc = ', rex_mfcc.shape)
print('Shape of rex_label = ', rex_label_new.shape)

# class other, augmentated
other_1 = np.load('audio_processing\Train_Data\set_complete_test_other_mfcc_cutout.npy')
other_2 = np.load('audio_processing\Train_Data\set_complete_test_other_mfcc_mixed_frequency_masking.npy')
other_3 = np.load('audio_processing\Train_Data\set_complete_test_other_mfcc_mixup.npy')
other_4 = np.load('audio_processing\Train_Data\set_complete_test_other_mfcc_sample_pairing.npy')
other_5 = np.load('audio_processing\Train_Data\set_complete_test_other_mfcc_specaugment.npy')
other_6 = np.load('audio_processing\Train_Data\set_complete_test_other_mfcc_specmix.npy')
other_7 = np.load('audio_processing\Train_Data\set_complete_test_other_mfcc_vh_mixup.npy')

# create other label array
# Combine all 'other' mfcc data
other_mfcc = np.concatenate((other_1, other_2, other_3, other_4, other_5, other_6, other_7), axis=0)

# Create a new label array the same size as other_mfcc.shape[0]
other_label = np.zeros((other_mfcc.shape[0], 9))  # assuming you have 9 classes
other_label[:, 8] = 1  # set the 9th position to 1 for all samples

# print shapes
print('Shape of other_mfcc = ', other_mfcc.shape)
print('Shape of other_label = ', other_label.shape)


# combine all data into one mfcc array and one label array
mfcc = np.concatenate((a_mfcc, b_mfcc, c_mfcc, one_mfcc, two_mfcc, three_mfcc, stopp_mfcc, rex_mfcc, other_mfcc), axis=0)
label = np.concatenate((a_label_new, b_label_new, c_label_new, one_label_new, two_label_new, three_label_new, stopp_label_new, rex_label_new, other_label), axis=0)

# print shapes
print('Shape of mfcc = ', mfcc.shape)
print('Shape of label = ', label.shape)

# randomize the order of the data
randomize = np.arange(mfcc.shape[0])
np.random.shuffle(randomize)
mfcc = mfcc[randomize]
label = label[randomize]

# print shapes
print('Shape of mfcc after randomizing = ', mfcc.shape)
print('Shape of label after randomizing = ', label.shape)

# save mfcc and label arrays
np.save('audio_processing\Train_Data\mfcc_complete_test.npy', mfcc)
np.save('audio_processing\Train_Data\label_complete_test.npy', label)

# plot class distribution
class_names = ['a', 'b', 'c', '1', '2', '3', 'stopp', 'rex', 'other']

# plot class distribution
plt.figure(figsize=(10, 10))
plt.hist(label.argmax(axis=1), bins=9)
plt.xticks(np.arange(9), class_names, rotation=45)
plt.show()

