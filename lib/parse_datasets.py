from lib.mulity_source import build_data_objects_for_test

#####################################################################################################
def parse_datasets(args, patch_ts=False, length_stat=False):

	device = args.device
	dataset_name = args.dataset

	##################################################################
	### PhysioNet dataset ### 
	### MIMIC dataset ###
	
	if dataset_name in ["mulity_source"]:
		data_objects = build_data_objects_for_test(args)
		return data_objects
	

	
	
