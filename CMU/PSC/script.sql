
-- Join tables

CREATE TABLE CustomerOrders AS
 SELECT 
    c.CustomerId,
    c.FirstName,
    c.LastName,
    c.AddressLine1 AS CustomerAddress1,
    c.AddressLine2 AS CustomerAddress2,
    c.City AS CustomerCity,
    c.State AS CustomerState,
    c.PostalCode AS CustomerPostalCode,
    c.Country AS CustomerCountry,
    c.CreditLimit,
    o.OrderId,
    o.Date AS OrderDate,
    o.DateRequired,
    o.DateShipped,
    o.Status AS OrderStatus,
    o.Comments AS OrderComments,
    od.ProductId,
    od.QuantityOrdered,
    od.PriceEach,
    od.OrderLineNumber,
    p.Line AS ProductLine,
    p.Size,
    p.Vendor AS ProductVendorName,
    p.Description AS ProductDescription,
    p.Inventory,
    p.BuyPrice,
    p.RetailPrice,
    p.Fabric,
    p.Color,
    p.Item,
    p.VendorId,
    l.Description AS LineDescription,
    v.Name AS VendorName,
    v.AddressLine1 AS VendorAddress1,
    v.AddressLine2 AS VendorAddress2,
    v.City AS VendorCity,
    v.State AS VendorState,
    v.PostalCode AS VendorPostalCode,
    v.Country AS VendorCountry,
    z.City AS ZipCity,
    z.StateName AS ZipState,
    z.Population AS ZipPopulation,
    z.timezone AS ZipTimezone,
    z.Lat as Lat,
    z.Lng as Lng
FROM Customer c
JOIN Orders o ON c.CustomerId = o.CustomerId
JOIN OrderDetail od ON o.OrderId = od.OrderId
JOIN Product p ON od.ProductId = p.ProductId
JOIN Line l ON p.Line = l.Line
JOIN Vendor v ON p.VendorId = v.VendorId
LEFT JOIN Zips z ON c.PostalCode = z.Zip;


-- Logic:
-- Locate HQ in zips table
-- compare each customer's location with hq via law of cosines to calculate distance (radius of earth is 6371)
-- Formula: R_earth * Arc cos(cos(hq_lat) * cos(order_lat) * cos (order_lng - hq_lng) + sin(hq_lat + odre_lat)
-- get number of customers per zip code
-- filter for zipcodes that are <= 100 km
-- get 3 most populated zipcodes
WITH hq_loc AS(SELECT Lat as hq_lat, Lng as hq_long FROM Zips WHERE zip = 02139), 
in_range AS (SELECT c.CustomerPostalCode,
           c.ZipCity,
           c.ZipState,
           COUNT(DISTINCT c.OrderId) AS total_orders,
           (6371 * ACOS(
                COS(RADIANS(hq_loc.hq_lat)) * COS(RADIANS(c.Lat)) *
                COS(RADIANS(c.Lng) - RADIANS(hq_loc.hq_long)) +
                SIN(RADIANS(hq_loc.hq_lat)) * SIN(RADIANS(c.Lat))
           )) AS distance
           FROM CustomerOrders c, hq_loc
           GROUP BY c.CustomerPostalCode, c.ZipCity, c.ZipState, c.Lat, c.Lng, hq_loc.hq_lat, hq_loc.hq_long
           HAVING distance <= 100)
SELECT CustomerPostalCode, ZipCity, ZipState, total_orders, ROUND(distance,3) AS distance
FROM in_range
ORDER BY total_orders DESC
LIMIT 3;