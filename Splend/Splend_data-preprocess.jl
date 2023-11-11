using DrWatson
@quickactivate "FitHMM-jl"

using CSV, DataFrames, Dates, ShiftedArrays, JLD2

function find_angle(lat1, long1, lat2, long2)   # tested
    lat1_r = lat1 .* pi ./ 180
    lat2_r = lat2 .* pi ./ 180
    # long1_r = long1 .* pi ./ 180
    # long2_r = long2 .* pi ./ 180
    delta_long = (long2 .- long1) .* pi ./ 180
    theta_r = atan.(sin.(delta_long).*cos.(lat2_r), 
                    cos.(lat1_r).*sin.(lat2_r) .- sin.(lat1_r).*cos.(lat2_r).*cos.(delta_long))
    
    return theta_r
end

function angle_change(unit, angle1, angle2) # tested
    if unit == "degree"
        change = angle2 .- angle1
        change = map(passmissing(x -> ifelse(abs(x) <= abs(x-360), x, x-360)), change)
        change = map(passmissing(x -> ifelse(x <= 180, x + 360, x)), change)
    else # unit == "radian"
        change = angle2 .- angle1
        change = map(passmissing(x -> ifelse(x <= -pi, x + 2*pi, ifelse(x >= pi, x - 2*pi, x))), change)
    end
    return change
end

function enum_trips_and_find_start(dataframe)   # tested
    sort!(dataframe, [:DateTime])
    TripId_list = DataFrame(TripId = unique(dataframe.TripId), ID = 1:length(unique(dataframe.TripId)))
    dataframe = innerjoin(dataframe, TripId_list, on = :TripId)
    dataframe.start = [1; dataframe.ID[2:end] - dataframe.ID[1:(end-1)]]
    return dataframe
end

function enum_hardware_trips_and_find_start(dataframe)   # tested
    sort!(dataframe, [:SubjectId, :TripId, :DateTime])
    group_df = groupby(dataframe, :SubjectId)
    result_df = combine(group_df) do sub_df
        enum_trips_and_find_start(sub_df)
    end
    return result_df
end

subject_df = CSV.read(datadir("Splend_Test/Splend_Test_subjectdf.csv"), DataFrame)
trip_df = CSV.read(datadir("Splend_Test/Splend_Test_triplistdf.csv"), DataFrame)
df = CSV.read(datadir("Splend_Test/Splend_Test_gps.csv"), DataFrame)

# matching between the 3 dataframes
subject_df.SubjectId = 1:nrow(subject_df)
subject_df.AssignmentStart = DateTime.(SubString.(subject_df.AssignmentStart, 1, 19), DateFormat("yyyy-mm-dd HH:MM:SS"))
trip_df.AssignmentStart = DateTime.(SubString.(trip_df.AssignmentStart, 1, 19), DateFormat("yyyy-mm-dd HH:MM:SS"))
trip_df = innerjoin(subject_df[!, [:HardwareId, :AssignmentStart, :SubjectId]], trip_df, 
                    on = [:HardwareId, :AssignmentStart])
df = innerjoin(trip_df[!, [:HardwareId, :TripId, :SubjectId]], df, on = [:HardwareId, :TripId])
df.DateTime = DateTime.(SubString.(df.DateTime, 1, 19), DateFormat("yyyy-mm-dd HH:MM:SS"))

# SubjectId + ID now uniquely defines a trip
df = enum_hardware_trips_and_find_start(df)
group_df = groupby(df, [:SubjectId, :ID])

transform!(group_df, :DateTime => (x -> Dates.value.(convert.(Dates.Second, x .- minimum(x)))) => :time_since)
transform!(group_df, :time_since => (x -> x - ShiftedArrays.lag(x)) => :time_interval)
df.time_interval = ifelse.(df.start .== 1, missing, df.time_interval)
filter!(row -> row.start .== 1 || row.time_interval .> 0, df)    # remove rows with 0 timeinterval first to ensure correct calculation of changes
group_df = groupby(df, [:SubjectId, :ID])

transform!(group_df, [:Latitude, :Longitude] => ((a,b) -> (find_angle(ShiftedArrays.lag(a), ShiftedArrays.lag(b), a, b))) => :radian)
df.radian = ifelse.(df.start .== 1, missing, df.radian)
transform!(group_df, :radian => (x -> angle_change("radian", ShiftedArrays.lag(x), x)) => :delta_radian)

transform!(group_df, [:Speed, :time_interval] => ((a, b) -> (a - ShiftedArrays.lag(a)) ./ b) => :acceleration)

filter_table = combine(groupby(df, [:SubjectId, :ID]), 
                        nrow => :num_obs,
                        :time_since => maximum => :trip_length,
                        :time_interval => (x -> minimum(skipmissing(x))) => :min_time_interval,
                        :time_interval => (x -> maximum(skipmissing(x))) => :max_time_interval)

# Consider trip length at least 180 sec (3 min)
# Consider num observations at least 30 for more meaningful evolution
id_filter = filter_table[filter_table.trip_length .>= 180 .&& filter_table.num_obs .>= 30, [:SubjectId, :ID]]

# Filter the original data
df_longer = innerjoin(df, id_filter, on = [:SubjectId, :ID])

jldsave(datadir("Splend_Test/df_longer.jld2"); df_longer = df_longer)
jldsave(datadir("Splend_Test/subject_df.jld2"); subject_df = subject_df)
jldsave(datadir("Splend_Test/trip_df.jld2"); trip_df = trip_df)

# subject_df.AssignmentEndMod = DateTime.(SubString.(subject_df.AssignmentEndMod, 1, 19), DateFormat("yyyy-mm-dd HH:MM:SS"))