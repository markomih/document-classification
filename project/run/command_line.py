import subprocess


class CommandLine:
    def __init__(self, interpreter_path, file_path, configuration):
        self.interpreter_path = interpreter_path
        self.file_path = file_path
        self.configuration = configuration

    @staticmethod
    def run_cmd(args_list, print_status, print_error):
        if print_status:
            print('Running system command: {0}'.format(' '.join(args_list)))

        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        std_output, std_err = proc.communicate()

        if print_status:
            print('Status {0}'.format(proc.returncode))

        if proc.returncode != 0 and print_error:
            print('ERROR\n', std_err.decode('utf-8'))

        if proc.returncode == 0:
            print("OUTPUT\n", std_output.decode('utf-8'))

        return proc.returncode, std_output.decode('utf-8'), std_err

    def run_configuration(self, print_status, print_error):
        CommandLine.run_cmd([
            self.interpreter_path, self.file_path,
            'with'] + self.configuration.get_params(), print_status, print_error)
