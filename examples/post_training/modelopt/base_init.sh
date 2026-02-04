function INIT_ENVS_CONFIG() {
    for item in `cat /proc/1/environ |tr '\0' '\n'`
    do
        export $item
    done
}


function UTILS_KILL {
    function killall {
        echo `ps -ef | grep $1 | grep -v grep | awk '{print $2}'`
        ps -ef | grep $1 |grep -v bin/tensorboard | grep -v jupyter-lab| grep -v grep | awk '{print $2}' |xargs kill -9
    }

    lsof -i:29500| grep -v grep | awk '{print $2}'|grep -v PID |xargs kill -9

    killall pretrain_.*.py
    killall /usr/bin/python
    killall log_period_monitor.py
    lsof -i:${MASTER_PORT}| grep -v grep | awk '{print $2}'|grep -v PID |xargs kill -9
}