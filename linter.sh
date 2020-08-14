# Run this script at project root by ".linter.sh" before you commit.
echo "Running isort..."
isort -sp .

echo "Running black..."
black -l 80 -t py38 .

echo "Running flake..."
flake8 .

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
