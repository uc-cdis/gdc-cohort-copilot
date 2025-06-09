#!/bin/bash
# Usage: ./split_tsv.sh /path/to/data.tsv

INPUT="$1"
DIR=$(dirname "$INPUT")
BASENAME=$(basename "$INPUT" .tsv)

# Get header
head -n 1 "$INPUT" > "$DIR/header.tsv"

# Get first 330k data rows (after header)
tail -n +2 "$INPUT" | head -n 330000 > "$DIR/part1.tsv"

# Get next 330k data rows
tail -n +2 "$INPUT" | head -n 660000 | tail -n 330000 > "$DIR/part2.tsv"

# Get final 340k data rows
tail -n +2 "$INPUT" | tail -n 340000 > "$DIR/part3.tsv"

# Combine header with each part
cat "$DIR/header.tsv" "$DIR/part1.tsv" > "$DIR/${BASENAME}_part1.tsv"
cat "$DIR/header.tsv" "$DIR/part2.tsv" > "$DIR/${BASENAME}_part2.tsv"
cat "$DIR/header.tsv" "$DIR/part3.tsv" > "$DIR/${BASENAME}_part3.tsv"

# Cleanup
rm "$DIR/header.tsv" "$DIR/part1.tsv" "$DIR/part2.tsv" "$DIR/part3.tsv"

echo "Done! Output: ${DIR}/${BASENAME}_part1.tsv, ${DIR}/${BASENAME}_part2.tsv, ${DIR}/${BASENAME}_part3.tsv"
