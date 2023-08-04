using DrWatson
@quickactivate "FitHMM-jl"

using CSV, DataFrames, Dates

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
        temp2 = change .- 360
        for i in eachindex(change)
            x = change[i]
            y = temp2[i]
            if !ismissing(x)    # not missing
                change[i] = ifelse(abs(x) <= abs(y), x, y)
                change[i] = ifelse(change[i] <= 180, change[i] + 360, change[i])
            end
        end
    else # unit == "radian"
        change = angle2 .- angle1
        for i in eachindex(change)
            x = change[i]
            if !ismissing(x)    # not missing
                change[i] = ifelse(x <= -pi, x + 2*pi, ifelse(x .>= pi, x .- 2*pi, x))
            end
        end
    end
    return change
end

function enum_trips_and_find_start(dataframe)   # tested
    dataframe = sort!(dataframe, [:DateTime])
    TripId_list = DataFrame(TripId = unique(dataframe.TripId), ID = 1:length(unique(dataframe.TripId)))
    dataframe = innerjoin(dataframe, TripId_list, on = :TripId)
    dataframe.start = [1; dataframe.ID[2:end] - dataframe.ID[1:(end-1)]]
    return dataframe
end


df = CSV.read(datadir("sample-data.csv"), DataFrame)
first(df, 10)

df.DateTime = DateTime.(SubString.(df.DateTime, 1, 19), DateFormat("yyyy-mm-dd HH:MM:SS"))

df = enum_trips_and_find_start(df)

df.radian = [missing; find_angle(df.Latitude[1:end-1], df.Longitude[1:end-1], df.Latitude[2:end], df.Longitude[2:end])]
df.radian = ifelse.(df.start .== 1, missing, df.radian)
df.delta_radian = [missing; angle_change("radian", df.radian[1:end-1], df.radian[2:end])]

df.timesince = combine(groupby(df, :ID), :DateTime => (x -> Dates.value.(convert.(Dates.Second, x .- minimum(x)))) => :timesince).timesince

df.timeinterval = [0; df.timesince[2:end] .- df.timesince[1:end-1]]
df.timeinterval = ifelse.(df.start .== 1, missing, df.timeinterval)

df.acceleration = [0; (df.Speed[2:end] - df.Speed[1:end-1]) ./ df.timeinterval[2:end]]


filter_table = combine(groupby(df, :ID), 
                        nrow => :num_obs,
                        :timesince => maximum => :trip_length,
                        :timeinterval => (x -> minimum(skipmissing(x))) => :min_timeinterval,
                        :timeinterval => (x -> maximum(skipmissing(x))) => :max_timeinterval)

# Consider trip length at least 180 sec (3 min)
# Consider num observations at least 30 for more meaningful evolution
id_filter = filter_table[filter_table.trip_length .>= 180 .& filter_table.num_obs .>= 30, :ID]

# Filter the original data
df_longer = filter(row -> row.ID in id_filter, df)
df_longer = filter(row -> row.start .== 1 || row.timeinterval .> 0, df_longer)