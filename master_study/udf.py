def string_run(node, generation_number):
	return f'''source {node.root.dictionary["setup_env_script"]}
cd {node.get_abs_path()}
python {node.root.dictionary["generations"][generation_number]["job_executable"]}'''
