# Task brainstorming
## Task 4:

Bin dep_times into segments to compare with mean taxi times by location
Note that the question requires knowledge of total flights at that time (how many flights occurred in a bin?)


## Task 5: What is the average percentage of arr_delays that is already created before wheels_off? (aka are arrival delays caused by departure delays?) Are airlines able to lower the delay during the flights?
 
Late arrival? 

How many flights depart late and arrive late? --> Use graph from task 2

is arrival delay < or > than departure delay? what are the percentages for this data?

## **Task 6**: How many states cover 50% of US air traffic? 

sample.groupby('origin_city_name').count().fl_date

Extract state from origin_city_name and arrival_city_name.
Compute % for each state, sort them high to low, and only return those that add up to 50

## **Task 7**: Test the hypothesis whether planes fly faster when there is the departure delay? 

- **air_time**: Flight Time, in Minutes
- **distance**: Distance between airports (miles)

--> Speed

- **dep_delay**:

Need to choose hypothesis test? ---> Try t_test and f_tests
Two groups: dep_delay <= 0 and dep_delay >0 vs speed on both groups

## **Task 8**: When (which hour) do most 'LONG', 'SHORT', 'MEDIUM' haul flights take off?
Short: air_time < 3hrs
Medium: airtime >= 3hrs and <6
Long: airtime >= 6

Create three categories as above and add column to DataFrame with category values. Bin wheels_off time into 24 segments and find mode() for each category.


## **Task 9**: Find the top 10 the bussiest airports. Does the biggest number of flights mean that the biggest number of passengers went through the particular airport? How much traffic do these 10 airports cover?

Group flights by traffic, and also by passengers.

Passengers: aggregate for entire year by airport. We'll need the sum of passengers that depart and passengers that arrive at *each* particular airport to create a total pax count.

FLights: aggregate for entire year by airport. Count total flights.
Once we have those two tables, create a single data frame with columns ['airport', 'pax', 'flights'] to compare, using 'airports' to join both tables.

Create composite columns (such as pax\*flights) for scoring metric.

