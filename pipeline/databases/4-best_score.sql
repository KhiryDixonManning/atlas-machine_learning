-- This script lists all records from second_table with a score >= 10.
-- Results display score and name, ordered by score descending.

SELECT score, name
FROM second_table
WHERE score >= 10
ORDER BY score DESC;-- This script computes the average score from the table second_table.
-- The result column is named average.
-- The database name is passed as an argument to the mysql command.

SELECT AVG(score) AS average FROM second_table;