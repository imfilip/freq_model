#!/bin/bash

set -e

source "$(cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")" && cd .. && pwd)"/bin/lib/vars.sh
source "$(cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")" && cd .. && pwd)"/bin/lib/fns.sh

# Script args
PYTHON_VERSION=${1:-"${DEFAULT_PYTHON_VERSION}"}
VENV_NAME=${1:-"${DEFAULT_VENV_NAME}"}
VENV_DIR=${1:-"${DEFAULT_VENV_DIR}"}
VENV_ID=${1:-"${DEFAULT_VENV_ID}"}

verify_python_version "${PYTHON_VERSION}"

PYTHON_INTERPRETER="python${PYTHON_VERSION}"
VENV_DIR_PATH="${VENV_DIR}/${VENV_NAME}"

echo "----------------------------------------"
echo "Python interpreter: ${PYTHON_INTERPRETER}"
echo "Virtual environment directory path: ${VENV_DIR_PATH}"
echo "----------------------------------------"

if [ ! -d "${VENV_DIR_PATH}" ]; then
    echo "Virtual env not found, creating one in ${VENV_DIR_PATH}..."
    ${PYTHON_INTERPRETER} -m venv "${VENV_DIR_PATH}" || exit 1
    
    # echo "Copy pip config to ${VENV_DIR_PATH}"
    # cp "${PROJECT_DIR}/requirements/pip.conf" "${VENV_DIR_PATH}" || exit 1
fi

echo ""
echo "--> Updating venv tools and installing packages..."
echo ""
"${VENV_DIR_PATH}/bin/pip3" install --no-cache-dir -U pip setuptools && \
"${VENV_DIR_PATH}/bin/pip3" install --no-cache-dir \
    -r "${PROJECT_DIR}/requirements/base.txt" \
    -r "${PROJECT_DIR}/requirements/extras.txt" || exit 1

echo ""
echo "--> Installing and configuring IPython kernel..."
echo ""
"${VENV_DIR_PATH}/bin/pip3" install --no-cache-dir ipykernel || exit 1

# Usunięcie starego kernela (jeśli istnieje)
echo "Removing old kernel if exists..."
"${VENV_DIR_PATH}/bin/jupyter" kernelspec uninstall -f "${VENV_NAME}" 2>/dev/null || true

# Instalacja kernela w środowisku wirtualnym
echo "Installing new kernel in virtual environment..."
"${VENV_DIR_PATH}/bin/python" -m ipykernel install \
    --prefix="${VENV_DIR_PATH}" \
    --name="${VENV_NAME}" \
    --display-name="Python (${VENV_NAME})" || exit 1

echo ""
echo "--> Checking environment setup..."
echo ""
"${VENV_DIR_PATH}/bin/python3" -c "import sys; print(f'Your Python version is: {sys.version}')" || exit 1

echo $PROJECT_DIR
echo "${PROJECT_DIR}" > "${VENV_DIR_PATH}/lib/${PYTHON_INTERPRETER}/site-packages/${VENV_ID}.pth" || exit 1
echo "--> Done." 

echo ""
echo "Jupyter kernel '${VENV_NAME}' has been installed successfully."
echo "You can now select this kernel in Jupyter notebooks." 