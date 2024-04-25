# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get if we're located in a docker and miniforge exists
FILE=/usr/local/DA_study/miniforge_docker/bin/activate

# Check if  the job is launched from docker
if [[ -f "$FILE" ]];

# Source
then
    source $FILE
    echo "Python activated from $FILE"
else
    source $SCRIPT_DIR/.venv/bin/activate
    echo "Python activated from $SCRIPT_DIR/.venv/bin/activate"
fi





