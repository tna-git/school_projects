-- Aggregate by college
SELECT College, COUNT(*) as Count
FROM world.fall_2023_uiuc
GROUP BY College;

-- Women in STEM Majors
SELECT College, SUM(Women) as Count
FROM world.fall_2023_uiuc
GROUP BY College;

-- Compare minority acceptances to majority
SELECT College, SUM(Caucasian+Asian_American) as Majority, 
	SUM(African_American+Hispanic+Native_American+Pacific_Isl+Multiracial) as Minority
FROM world.fall_2023_uiuc
GROUP BY College;

-- Count Niche Concentrations
SELECT College, COUNT(Concentration_Name) as Count
FROM world.fall_2023_uiuc
WHERE Concentration_Name != "" and Concentration_Name IS NOT NULL
GROUP BY College;

-- Number of IS Undergraduates
SELECT Degree, College, Total
FROM world.fall_2023_uiuc
WHERE Degree ='BS' and College = "Ischool";