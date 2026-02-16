
[ $# -eq 0 ] && echo Need message && exit

git add .

git commit -m "$1"

git push
