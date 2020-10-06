# This scrips add all changes to git updating the version of the changed files by one. And then commits using the message as the second parameter.
# Usage: ./auto-version-commit.sh "My commit message"

if [ "$1" != '' ]; then
IFS="
"
git status -s
for p in `git status -s`	# For each file in git status
do
	file=`echo "$p" | rev | cut -d ' ' -f1 | rev`		# Take the name of the file
	if [ "$file" != "auto-version-commit.sh" ]; then	# Ignores himself
		if grep -q "__version__ = " $file; then		# Check version is in the file
			echo "Increasing version of file $file"
			oldver=`grep "__version__ = " $file`	# oldver=__version__ = 'z.y.x'
			verf1=`echo $oldver | cut -d '.' -f1`	# __version__ = 'z
			verf2=`echo $oldver | cut -d '.' -f2`	# y
			verf3=`echo $oldver | cut -d '.' -f3`	# x'
			oldnum=`echo $verf3 | sed 's/.$//'`	# removes the quote from verf3
			newnum=`expr $oldnum + 1`		# x += 1
			newver="$verf1.$verf2.$newnum\'"	# joins string again.
			sed -i "s/$oldver\$/$newver/g" $file	# replaces line with the new one
		fi
	fi
done
file="cvnn/_version.py"
echo "Increasing version of file $file"
oldver=`grep "__version__ = " $file`	# oldver=__version__ = 'z.y.x'
verf1=`echo $oldver | cut -d '.' -f1`	# __version__ = 'z
verf2=`echo $oldver | cut -d '.' -f2`	# y
verf3=`echo $oldver | cut -d '.' -f3`	# x'
oldnum=`echo $verf3 | sed 's/.$//'`	# removes the quote from verf3
newnum=`expr $oldnum + 1`		# x += 1
newver="$verf1.$verf2.$newnum\'"	# joins string again.
sed -i "s/$oldver\$/$newver/g" $file	# replaces line with the new one

# Change doc version and date
file="docs/index.rst"
echo "Increasing version of file $file"
chang=`grep ":Version: " $file`
verf1=`echo $chang | cut -d '.' -f1`
dat=$(date +'%m/%d/%Y')
newlin="$verf1.$verf2.$newnum of $dat"
sed -i "s,$chang,$newlin,g" $file

git add -A
git commit -m $1
tagver="${verf1: -1}.$verf2.$newnum"
git tag $tagver
else
echo No commit message found, please add a message.
fi


	


