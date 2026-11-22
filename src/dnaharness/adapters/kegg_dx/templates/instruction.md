# Which disease does this variant contribute to?

The inputs for this question are in the JSON file `/task/inputs.json`. It contains:
- `question`: the question text (chromosome, pathway network, genes in the pathway, and the question itself).
{input_lines}
Work out which disease this variant contributes to.
{tool_lines}
The answer must be exactly one of these disease names, copied verbatim:
{label_lines}

When you are done, write ONLY the disease name to `/logs/agent/answer.txt`, for example:

    echo "parkinson's disease" > /logs/agent/answer.txt

Then finish.
