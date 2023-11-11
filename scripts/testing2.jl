using DrWatson
@quickactivate "FitHMM-jl"

using CSV, DataFrames, Statistics, Dates, Pipe

# functions 1)find_angle 2)angle_change 3)lag

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

function lag(x)
    return [missing; x[1:end-1]]
end


df = CSV.read(datadir("sample-data.csv"), DataFrame)
df.DateTime = DateTime.(SubString.(df.DateTime, 1, 19), DateFormat("yyyy-mm-dd HH:MM:SS"))
sort!(df, [:TripId, :DateTime]) 
first(df, 10) 

gdf = groupby(df,:TripId)
transform!(gdf, groupindices => :ID, eachindex => :start) # gdf.start = ifelse.(gdf[!,:start].== 1, 1, 0)

transform!(gdf, [:Latitude, :Longitude] => ((a,b) -> (find_angle(lag(a), lag(b), a, b))) => :radian)
transform!(gdf, :radian => (x -> angle_change("radian", lag(x), x)) => :delta_radian)

transform!(gdf, :DateTime => (x -> Dates.value.(convert.(Dates.Second, x .- minimum(x)))) => :timesince)

gdf = transform(gdf, :timesince => (x -> x - lag(x)) => :timeinterval)
replace!(gdf.timeinterval, missing => 0)

gdf = groupby(gdf, :ID)
gdf = transform(gdf, [:Speed,:timeinterval] => ((a, b) -> (a - lag(a))./b) => :acceleration)
replace!(gdf.acceleration, missing => 0.0)

filter_table = combine(groupby(gdf, :ID), 
                        nrow => :num_obs,
                        :timesince => maximum => :trip_length,
                        :timeinterval => (x -> minimum(skipmissing(x))) => :min_timeinterval,
                        :timeinterval => (x -> maximum(skipmissing(x))) => :max_timeinterval)

# Consider trip length at least 180 sec (3 min)
# Consider num observations at least 30 for more meaningful evolution
id_filter = filter_table[filter_table.trip_length .>= 180 .& filter_table.num_obs .>= 30, :ID]

# Filter the original data
gdf_longer = filter(row -> row.ID in id_filter, gdf)
gdf_longer = filter(row -> row.start .== 1 || row.timeinterval .> 0, gdf_longer)