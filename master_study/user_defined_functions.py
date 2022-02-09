def generate_run_sh(node, generation_number):
   python_command =  (node
                     .root
                     .parameters["generations"]
                     [generation_number]
                     ["job_executable"])
   return (f'source {node.root.parameters["setup_env_script"]}\n'
           f'cd {node.get_abs_path()}\n'
           f'python {python_command} > output.txt 2> error.txt\n')
