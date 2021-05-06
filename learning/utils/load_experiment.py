import os
from learning.policies.dqn import DQNAgent
from learning.utils import misc


class LoadExperiment:
    def __init__(self, experiment=None):
        """

        Args:
            backup:
        """
        self.repo = os.path.join(misc.RESOURCE_ROOT)
        self.experiment = experiment

    def list_names_in_experiment(self, return_paths=True):
        assert self.experiment is not None, 'No experiment name specified'
        path = os.path.join(self.repo, 'models', self.experiment)
        dirs = os.listdir(path)
        if return_paths:
            return_res = []
            for d in dirs:
                full_path = os.path.join(path, d)
                return_res.append(full_path)
        else:
            return_res = dirs
        return return_res

    def list_experiment_checkpoints(self, log_number):
        """

        Args:
            name:
            log_num:
            returntype:

        Returns:

        """
        names_in_experiment = self.list_names_in_experiment(return_paths=True)
        all_checkpoints = []
        for name in names_in_experiment:
            logs = sorted(os.listdir(name))
            path_to_checkpoints = os.path.join(name, logs[log_number], 'checkpoints')
            assert os.path.exists(path_to_checkpoints)

            checkpoints = sorted([int(c) for c in os.listdir(path_to_checkpoints)])
            all_checkpoints = all_checkpoints + [os.path.join(path_to_checkpoints, str(checkpoint)) for checkpoint in checkpoints]

        return all_checkpoints

    def list_logs(self, name, return_type='paths'):
        """

        Args:
            name:
            return_type:

        Returns:

        """
        log_path = os.path.join(self.repo, 'models', name)
        logs = os.listdir(log_path)
        if return_type == 'paths':
            ret = [os.path.join(log_path, log) for log in logs]
        else:
            ret = logs
        return sorted(ret)

    def list_checkpoints(self, name, log_num, returntype='paths'):
        """

        Args:
            name:
            log_num:
            returntype:

        Returns:

        """
        checkpoint_path = os.path.join(self.repo, 'models', name, self.list_logs(name)[log_num], 'checkpoints')
        assert os.path.exists(checkpoint_path)

        checkpoints = sorted([int(c) for c in os.listdir(checkpoint_path)])
        if returntype == 'paths':
            ret = [os.path.join(checkpoint_path, str(checkpoint)) for checkpoint in checkpoints]
        else:
            ret = checkpoints
        return  ret

    def load_agent(self, name, lognumber=0):
        """

        Args:
            name:
            lognumber:

        Returns:

        """
        log_paths = self.list_logs(name)
        model_path = log_paths[lognumber]
        agent = DQNAgent.load(model_path)
        agent.main_net.compile(loss='MSE')
        return agent

    def load_agent_from_path(self, model_path):
        """

        Args:
            model_path:

        Returns:

        """
        agent = DQNAgent.load(model_path)
        agent.main_net.compile(loss='MSE')
        return agent


if __name__ == '__main__':
    ldr = LoadExperiment('honiklada')
    print(ldr.list_names_in_experiment())
    cps = ldr.list_experiment_checkpoints(log_number=0)
    print(cps)
    ag = ldr.load_agent_from_path(cps[1])

    # print(ldr.list_logs('DQN-0'))
    # print(ldr.list_checkpoints('DQN-0', 0))
    #
    # cps = ldr.list_checkpoints('MONOSPLINE-0', 0)
    # ag = ldr.load_agent_from_path(cps[3])