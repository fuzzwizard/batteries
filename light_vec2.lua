--[[
	2d vector type that uses array incidies rather than x, y.
	better performance characteristics when used en-masse.
	api-compatible with vec2
]]

local path = (...):gsub('light_vec2', '')
local class = require(path .. 'class')
local math = require(path .. 'mathx') --shadow global math module
local make_pooled = require(path .. 'make_pooled')

local light_vec2 = class({
	name = 'light_vec2',
})

--stringification
function light_vec2:__tostring()
	return ('(%.2f, %.2f)'):format(self[1], self[2])
end

--ctor
function light_vec2:new(x, y)
	if type(x) == 'number' then
		self:scalar_set(x, y)
	elseif type(x) == 'table' then
		if type(x.type) == 'function' and x:type() == 'light_vec2' then
			self:vector_set(x)
		elseif x[1] then
			self:scalar_set(x[1], x[2])
		else
			self:scalar_set(x.x, x.y)
		end
	else
		self:scalar_set(0)
	end
end

--explicit ctors; mostly vestigial at this point
function light_vec2:copy()
	return light_vec2(self[1], self[2])
end

function light_vec2:x()
	return self[1]
end

function light_vec2:y()
	return self[2]
end

function light_vec2:polar(length, angle)
	return light_vec2(length, 0):rotate_inplace(angle)
end

function light_vec2:filled(v)
	return light_vec2(v, v)
end

function light_vec2:zero()
	return light_vec2(0)
end

--unpack for multi-args
function light_vec2:unpack()
	return self[1], self[2]
end

--pack when a sequence is needed
function light_vec2:pack()
	return { self:unpack() }
end

--shared pooled storage
make_pooled(light_vec2, 128)

--get a pooled copy of an existing vector
function light_vec2:pooled_copy()
	return light_vec2:pooled(self)
end

--modify

function light_vec2:vector_set(v)
	self[1] = v[1]
	self[2] = v[2]
	return self
end

function light_vec2:scalar_set(x, y)
	if not y then y = x end
	self[1] = x
	self[2] = y
	return self
end

function light_vec2:swap(v)
	local sx, sy = self[1], self[2]
	self[1], self[2] = v[1], v[2]
	v[1], v[2] = sx, sy
	return self
end

-----------------------------------------------------------
--equality comparison
-----------------------------------------------------------

--threshold for equality in each dimension
local EQUALS_EPSILON = 1e-9

--true if a and b are functionally equivalent
function light_vec2.equals(a, b)
	return (
		math.abs(a[1] - b[1]) <= EQUALS_EPSILON and
		math.abs(a[2] - b[2]) <= EQUALS_EPSILON
	)
end

--true if a and b are not functionally equivalent
--(very slightly faster than `not light_vec2.equals(a, b)`)
function light_vec2.nequals(a, b)
	return (
		math.abs(a[1] - b[1]) > EQUALS_EPSILON or
		math.abs(a[2] - b[2]) > EQUALS_EPSILON
	)
end

--alias
light_vec2.not_equals = light_vec2.nequals

-----------------------------------------------------------
--arithmetic
-----------------------------------------------------------

--vector
function light_vec2:vector_add_inplace(v)
	self[1] = self[1] + v[1]
	self[2] = self[2] + v[2]
	return self
end

function light_vec2:vector_sub_inplace(v)
	self[1] = self[1] - v[1]
	self[2] = self[2] - v[2]
	return self
end

function light_vec2:vector_mul_inplace(v)
	self[1] = self[1] * v[1]
	self[2] = self[2] * v[2]
	return self
end

function light_vec2:vector_div_inplace(v)
	self[1] = self[1] / v[1]
	self[2] = self[2] / v[2]
	return self
end

--(a + (b * t))
--useful for integrating physics and adding directional offsets
function light_vec2:fused_multiply_add_inplace(v, t)
	self[1] = self[1] + (v[1] * t)
	self[2] = self[2] + (v[2] * t)
	return self
end

--scalar
function light_vec2:scalar_add_inplace(x, y)
	if not y then y = x end
	self[1] = self[1] + x
	self[2] = self[2] + y
	return self
end

function light_vec2:scalar_sub_inplace(x, y)
	if not y then y = x end
	self[1] = self[1] - x
	self[2] = self[2] - y
	return self
end

function light_vec2:scalar_mul_inplace(x, y)
	if not y then y = x end
	self[1] = self[1] * x
	self[2] = self[2] * y
	return self
end

function light_vec2:scalar_div_inplace(x, y)
	if not y then y = x end
	self[1] = self[1] / x
	self[2] = self[2] / y
	return self
end

-----------------------------------------------------------
-- geometric methods
-----------------------------------------------------------

function light_vec2:length_squared()
	return self[1] * self[1] + self[2] * self[2]
end

function light_vec2:length()
	return math.sqrt(self:length_squared())
end

function light_vec2:distance_squared(other)
	local dx = self[1] - other[1]
	local dy = self[2] - other[2]
	return dx * dx + dy * dy
end

function light_vec2:distance(other)
	return math.sqrt(self:distance_squared(other))
end

function light_vec2:normalise_both_inplace()
	local len = self:length()
	if len == 0 then
		return self, 0
	end
	return self:scalar_div_inplace(len), len
end

function light_vec2:normalise_inplace()
	local v, len = self:normalise_both_inplace()
	return v
end

function light_vec2:normalise_len_inplace()
	local v, len = self:normalise_both_inplace()
	return len
end

function light_vec2:inverse_inplace()
	return self:scalar_mul_inplace(-1)
end

-- angle/direction specific

function light_vec2:rotate_inplace(angle)
	local s = math.sin(angle)
	local c = math.cos(angle)
	local ox = self[1]
	local oy = self[2]
	self[1] = c * ox - s * oy
	self[2] = s * ox + c * oy
	return self
end

function light_vec2:rotate_around_inplace(angle, pivot)
	self:vector_sub_inplace(pivot)
	self:rotate_inplace(angle)
	self:vector_add_inplace(pivot)
	return self
end

--fast quarter/half rotations
function light_vec2:rot90r_inplace()
	local ox = self[1]
	local oy = self[2]
	self[1] = -oy
	self[2] = ox
	return self
end

function light_vec2:rot90l_inplace()
	local ox = self[1]
	local oy = self[2]
	self[1] = oy
	self[2] = -ox
	return self
end

light_vec2.rot180_inplace = light_vec2.inverse_inplace --alias

--get the angle of this vector relative to (1, 0)
function light_vec2:angle()
	return math.atan2(self[2], self[1])
end

--get the normalised difference in angle between two vectors
function light_vec2:angle_difference(v)
	return math.angle_difference(self:angle(), v:angle())
end

--lerp towards the direction of a provided vector
--(length unchanged)
function light_vec2:lerp_direction_inplace(v, t)
	return self:rotate_inplace(self:angle_difference(v) * t)
end

-----------------------------------------------------------
-- per-component clamping ops
-----------------------------------------------------------

function light_vec2:min_inplace(v)
	self[1] = math.min(self[1], v[1])
	self[2] = math.min(self[2], v[2])
	return self
end

function light_vec2:max_inplace(v)
	self[1] = math.max(self[1], v[1])
	self[2] = math.max(self[2], v[2])
	return self
end

function light_vec2:clamp_inplace(min, max)
	self[1] = math.clamp(self[1], min[1], max[1])
	self[2] = math.clamp(self[2], min[2], max[2])
	return self
end

-----------------------------------------------------------
-- absolute value
-----------------------------------------------------------

function light_vec2:abs_inplace()
	self[1] = math.abs(self[1])
	self[2] = math.abs(self[2])
	return self
end

-----------------------------------------------------------
-- sign
-----------------------------------------------------------

function light_vec2:sign_inplace()
	self[1] = math.sign(self[1])
	self[2] = math.sign(self[2])
	return self
end

-----------------------------------------------------------
-- truncation/rounding
-----------------------------------------------------------

function light_vec2:floor_inplace()
	self[1] = math.floor(self[1])
	self[2] = math.floor(self[2])
	return self
end

function light_vec2:ceil_inplace()
	self[1] = math.ceil(self[1])
	self[2] = math.ceil(self[2])
	return self
end

function light_vec2:round_inplace()
	self[1] = math.round(self[1])
	self[2] = math.round(self[2])
	return self
end

-----------------------------------------------------------
-- interpolation
-----------------------------------------------------------

function light_vec2:lerp_inplace(other, amount)
	self[1] = math.lerp(self[1], other[1], amount)
	self[2] = math.lerp(self[2], other[2], amount)
	return self
end

function light_vec2:lerp_eps_inplace(other, amount, eps)
	self[1] = math.lerp_eps(self[1], other[1], amount, eps)
	self[2] = math.lerp_eps(self[2], other[2], amount, eps)
	return self
end

-----------------------------------------------------------
-- vector products and projections
-----------------------------------------------------------

function light_vec2:dot(other)
	return self[1] * other[1] + self[2] * other[2]
end

--"fake", but useful - also called the wedge product apparently
function light_vec2:cross(other)
	return self[1] * other[2] - self[2] * other[1]
end

function light_vec2:scalar_projection(other)
	local len = other:length()
	if len == 0 then
		return 0
	end
	return self:dot(other) / len
end

function light_vec2:vector_projection_inplace(other)
	local div = other:dot(other)
	if div == 0 then
		return self:scalar_set(0)
	end
	local fac = self:dot(other) / div
	return self:vector_set(other):scalar_mul_inplace(fac)
end

function light_vec2:vector_rejection_inplace(other)
	local tx, ty = self[1], self[2]
	self:vector_projection_inplace(other)
	self:scalar_set(tx - self[1], ty - self[2])
	return self
end

--get the winding side of p, relative to the line a-b
-- (this is based on the signed area of the triangle a-b-p)
-- return value:
--	>0 when p left of line
--	=0 when p on line
--	<0 when p right of line
function light_vec2.winding_side(a, b, p)
	return (b[1] - a[1]) * (p[2] - a[2])
		- (p[1] - a[1]) * (b[2] - a[2])
end

--return whether a is nearer to v than b
function light_vec2.nearer(v, a, b)
	return v:distance_squared(a) < v:distance_squared(b)
end

-----------------------------------------------------------
-- vector extension methods for special purposes
--   (any common vector ops worth naming)
-----------------------------------------------------------

--"physical" friction
function light_vec2:apply_friction_inplace(mu, dt)
	local friction = self:pooled_copy():scalar_mul_inplace(mu * dt)
	if friction:length_squared() > self:length_squared() then
		self:scalar_set(0, 0)
	else
		self:vector_sub_inplace(friction)
	end
	friction:release()
	return self
end

--"gamey" friction in one dimension
local function _friction_1d(v, mu, dt)
	local friction = mu * v * dt
	if math.abs(friction) > math.abs(v) then
		return 0
	else
		return v - friction
	end
end

--"gamey" friction in both dimensions
function light_vec2:apply_friction_xy_inplace(mu_x, mu_y, dt)
	self[1] = _friction_1d(self[1], mu_x, dt)
	self[2] = _friction_1d(self[2], mu_y, dt)
	return self
end

--minimum/maximum components
function light_vec2:mincomp()
	return math.min(self[1], self[2])
end

function light_vec2:maxcomp()
	return math.max(self[1], self[2])
end

-- meta functions for mathmatical operations
function light_vec2.__add(a, b)
	return a:vector_add(b)
end

function light_vec2.__sub(a, b)
	return a:vector_sub(b)
end

function light_vec2.__mul(a, b)
	if type(a) == 'number' then
		return b:scalar_mul(a)
	elseif type(b) == 'number' then
		return a:scalar_mul(b)
	else
		return a:vector_mul(b)
	end
end

function light_vec2.__div(a, b)
	if type(b) == 'number' then
		return a:scalar_div(b)
	else
		return a:vector_div(b)
	end
end

-- mask out min component, with preference to keep x
function light_vec2:major_inplace()
	if self[1] > self[2] then
		self[2] = 0
	else
		self[1] = 0
	end
	return self
end

-- mask out max component, with preference to keep x
function light_vec2:minor_inplace()
	if self[1] < self[2] then
		self[2] = 0
	else
		self[1] = 0
	end
	return self
end

--vector_ free alias; we're a vector library, so semantics should default to vector
light_vec2.add_inplace = light_vec2.vector_add_inplace
light_vec2.sub_inplace = light_vec2.vector_sub_inplace
light_vec2.mul_inplace = light_vec2.vector_mul_inplace
light_vec2.div_inplace = light_vec2.vector_div_inplace
light_vec2.set = light_vec2.vector_set

--american spelling alias
light_vec2.normalize_both_inplace = light_vec2.normalise_both_inplace
light_vec2.normalize_inplace = light_vec2.normalise_inplace
light_vec2.normalize_len_inplace = light_vec2.normalise_len_inplace

--garbage generating functions that return a new vector rather than modifying self
for _, inplace_name in ipairs({
	'vector_add_inplace',
	'vector_sub_inplace',
	'vector_mul_inplace',
	'vector_div_inplace',
	'fused_multiply_add_inplace',
	'add_inplace',
	'sub_inplace',
	'mul_inplace',
	'div_inplace',
	'scalar_add_inplace',
	'scalar_sub_inplace',
	'scalar_mul_inplace',
	'scalar_div_inplace',
	'normalise_both_inplace',
	'normalise_inplace',
	'normalise_len_inplace',
	'normalize_both_inplace',
	'normalize_inplace',
	'normalize_len_inplace',
	'inverse_inplace',
	'rotate_inplace',
	'rotate_around_inplace',
	'rot90r_inplace',
	'rot90l_inplace',
	'lerp_direction_inplace',
	'min_inplace',
	'max_inplace',
	'clamp_inplace',
	'abs_inplace',
	'sign_inplace',
	'floor_inplace',
	'ceil_inplace',
	'round_inplace',
	'lerp_inplace',
	'lerp_eps_inplace',
	'vector_projection_inplace',
	'vector_rejection_inplace',
	'apply_friction_inplace',
	'apply_friction_xy_inplace',
	'major_inplace',
	'minor_inplace',
}) do
	local garbage_name = inplace_name:gsub('_inplace', '')
	light_vec2[garbage_name] = function(self, ...)
		self = self:copy()
		return self[inplace_name](self, ...)
	end
end

--"hungarian" shorthand aliases for compatibility and short names
--
--i do encourage using the longer versions above as it makes code easier
--to understand when you come back, but i also appreciate wanting short code
for _, v in ipairs({
	{ 'sset',      'scalar_set' },
	{ 'sadd',      'scalar_add' },
	{ 'ssub',      'scalar_sub' },
	{ 'smul',      'scalar_mul' },
	{ 'sdiv',      'scalar_div' },
	{ 'vset',      'vector_set' },
	{ 'vadd',      'vector_add' },
	{ 'vsub',      'vector_sub' },
	{ 'vmul',      'vector_mul' },
	{ 'vdiv',      'vector_div' },
	--(no plain addi etc, imo it's worth differentiating vaddi vs saddi)
	{ 'fma',       'fused_multiply_add' },
	{ 'vproj',     'vector_projection' },
	{ 'vrej',      'vector_rejection' },
	--just for the _inplace -> i shorthand, mostly for backwards compatibility
	{ 'min',       'min' },
	{ 'max',       'max' },
	{ 'clamp',     'clamp' },
	{ 'abs',       'abs' },
	{ 'sign',      'sign' },
	{ 'floor',     'floor' },
	{ 'ceil',      'ceil' },
	{ 'round',     'round' },
	{ 'lerp',      'lerp' },
	{ 'rotate',    'rotate' },
	{ 'normalise', 'normalise' },
	{ 'normalize', 'normalize' },
}) do
	local shorthand, original = v[1], v[2]
	if light_vec2[shorthand] == nil then
		light_vec2[shorthand] = light_vec2[original]
	end
	--and inplace version
	shorthand = shorthand .. 'i'
	original = original .. '_inplace'
	if light_vec2[shorthand] == nil then
		light_vec2[shorthand] = light_vec2[original]
	end
end

return light_vec2
