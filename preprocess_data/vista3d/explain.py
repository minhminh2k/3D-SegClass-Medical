# In this case, user need to map the index in the original dataset label into the global class mapping defined in label_dict.
# "1" represents spleen in original label, while "3" represents spleen as defined by label_dict. If the target organ is not
# found in label_dict, user can use any value smaller than 512 (recommend using < 255 since Relabeld transform use torch.unit8).
# Relabeld transform will map "1" to "3" in the groundtruth label. if "2" is liver in this dataset ("1" in label_dict, the mapping is [[1,3],[2,1]]

# label_set represents the class indexes after Relabeld that will be included during training. For finetuning, 0 can be included but not necessary.
