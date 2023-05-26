:: Insert the name of the dam failure runs file 
:: (include path if not in the same folder as this batch file)
set dam_failure_runs="DamFailureRuns.json"

:: Insert the name of the python file
:: (include path if not in the same folder as this batch file)
set pyfile="..\main.py"

:: Activate the Anaconda base environment (if using conda)
call C:\Users\sharper\Anaconda3\condabin\activate.bat

:: Run the model
python %pyfile% %dam_failure_runs%

pause